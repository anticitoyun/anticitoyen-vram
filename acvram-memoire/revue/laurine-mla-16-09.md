# Laurine — chantier MLA (ordre Sage `sage-duel-verdict-16-09` § 3 et § 6), 16/09

## Commit 1 — chemin MLA du décodage batché au lot b (ce commit)
Constat (nsys de Laure, § 6.1) : 15 534 lancements/pas à b=12 (Coder : 1 517), dont 2 845
`int8_gemv<4,1>` à lot 1, ~5 000 cat/copies, 12 rmsnorm par couche — le chemin MLA tournait
créneau par créneau (`ACVRAM_MLA_BATCH=1` : `decode_static_batch`, préparation et sortie par
créneau, seule l'attention batchée).

Fait :
- `ACVRAM_MLA_BATCH` défaut 1 → **2** : `decode_static_batch_complet` (ab9dbca, à sec depuis le
  13/09, test `test_module_batch_complet` jamais passé sur carte) devient le chemin — projections
  en un GEMM M=b, normes, RoPE, cat, einsum, o_proj sur le lot (`mla.py`).
- Ce qui restait par créneau dans ce chemin — 12 `index_copy_` + 12 `len.add_` par couche
  (1 104 lancements/pas) — passe en UN lancement : `mla_ecrit_latent(k_new [B,W], cache_ptrs,
  len_ptrs)` (`acvram_kernels.cu`, un bloc par créneau : cache_b[len_b] = k_new[b], len_b += 1) ;
  `_mla_lot` (`model.py`) porte maintenant la table des adresses des longueurs (clé = adresses
  des caches et des longueurs, construite à l'échauffement, hors capture).
- `v_b` en fp32 converti une fois (`_v_b32`) au lieu d'une conversion par couche et par pas.
- Tests : `test_module_batch_complet` avec `len_ptrs` (bit-identique à `forward_batch`, ≤ 1 ulp
  de la boucle) ; `test_ecrit_latent_un_lancement` (noyau == boucle index_copy_/add_ ; le nombre
  de lancements du chemin complet est le même à B=4 et B=12 — casse si une boucle par créneau
  revient). Non exécutés sur carte.

Prédiction scellée avant mesure (Laure, même nsys) : par couche MLA ≈ 30 lancements (2 GEMM,
2 normes, RoPE ≈ 10, cat ×3, einsum ×2 ≈ 4, écriture 1, attention 2, o_proj 1, conversions 2)
→ 46 × 30 ≈ 1 400 + le reste du pas (MoE, denses, ≈ 1 000) = **2 200-2 600 lancements/pas** ;
scellé Sage ≤ 2 500 — si > 2 500, la marche suivante est nommée : RoPE + cat en un noyau
(≈ −10/couche). Pas b=12 : scellé ≤ 22 ms (réfuté > 26 → trou hôte, chantier suivant) ;
mon attendu 24-28 ms (les 5,6 ms de GEMV lot 1 et ~8 ms de cat/copies tombent, le trou de
5,3 ms reste). Équivalence : top-1 identique, cos ≥ 0,9999 contre `ACVRAM_MLA_BATCH=1` sur
16 positions × 2 couches ; PPL 3 tranches ± 0,004.

Verdict Laure (3f69c82) : TENU — 21,44 ms (témoin 43,66), 2 541 lancements (41 au-dessus de
2 500), logits bit-identiques 768/768, PPL B/A = 1,000000.

## Marche « RoPE + cat » (commit 1b) : préparation du lot en un noyau
`mla_prep_batch(q, kvp, lens, cos32, sin32, k_b, w_norm, …) → (q_eff fp32 [B,nh,W], k_new bf16
[B,W])` remplace par couche : RoPE (≈ 10 élémentaires), trois `cat`, l'einsum k_b (bmm +
copies), la norme kv_a, la conversion fp32 — ≈ 15 lancements → 1. Arithmétique : RoPE
opération par opération comme torch (cos/sin bf16, produits et somme arrondis bf16), norme =
la réduction de `rmsnorm_bf16_kernel` à l'identique (k_new bit-identique), q_abs accumulé en
fp32 puis arrondi bf16 (≤ 1 ulp de cuBLAS). Témoin `ACVRAM_MLA_PREP_NOYAU=0`. Tests :
`test_module_batch_complet` (témoin bit-identique à forward_batch ; noyau ≤ 1 ulp de la boucle,
caches identiques), `test_prep_batch_avec_rope` (k_new bit-identique, y ≤ 1 ulp).
Prédiction : 2 541 → **≈ 1 850 lancements/pas** (−15 × 46), pas 21,4 → 19,5-20,5 ms.

## Commit 2 — attention à une passe (prêt sur `laurine-mla-1p`, 0309a03)
`mla_decode_1p` : le cache latent lu une fois pour les H têtes (tuile de 32 lignes en
shared, scores fp32 des H têtes — pas de mma bf16 sur q : 2⁻⁸ sur des scores ~10 aurait coûté
le cos ≥ 0,9999 —, softmax en ligne, o_lat en registres, tranches de L recombinées) ;
`ACVRAM_MLA_UNE_PASSE=0` rejoue les deux noyaux. Scellé : mla_* 6,4 → ≤ 3 ms. Fusionné sur
laurine après le verdict du commit 1.

## Commit 3 — latent fp8 par ligne (`sage-avis-exterieur` § 6)
Écrit, **défaut OFF** (`ACVRAM_MLA_LATENT_FP8=1` pour l'essai) : le cache des créneaux devient
`[max_len, W+16] uint8` (W codes E4M3 + échelle fp32 s = amax/448 par ligne) ; `new_static`,
`static_load` (quantifie), `static_export` (déquantifie), `_prep_decode` (torch) et
`mla_ecrit_latent(…, fp8)` (noyau, bit-identique à `_fp8_quant_rows`) écrivent ce format ;
`mla_decode_1p(…, fp8)` le lit : codes en shared, table E4M3→float, échelle de ligne appliquée
au score et absorbée dans p_r — moitié des octets lus, même arithmétique fp32 ensuite (test :
fp8 sur les codes == bf16 sur le déquantifié à 1e-5). Les deux noyaux d'avant restent bf16
(fp8 force le chemin 1p). Attendu (Sage) : ≤ 0,5 ms de gain sur mla_* ; **réfuté si PPL 3
tranches sort de ± 0,004** → bf16 gardé. Coût VRAM : −50 % sur le latent des créneaux.
Prédiction : mla_* −0,2 à −0,5 ms ; PPL : E4M3 par ligne = 3 bits de mantisse sur un latent
normé, je n'exclus pas un dépassement (vLLM KV fp8 : +5 points) — c'est la mesure qui dit.

## Après le duel (Sage § 14) : deux postes sans coût de PPL
- **(ii) tête** (54d5b28) : `int8_gemv(…, sortie_fp32=True)` = noyau `<bf16, float>` — x reste
  en bf16, accumulation et sortie fp32 : mêmes produits, même ordre de sommes que `x.to(float32)`
  → logits **égaux au bit** (test `test_sortie_fp32_egale_au_bit_au_chemin_x_fp32`, 151 936 ×
  2 048, N = 1/12/16 ; le chemin bf16 diverge). `MoEModel._tete` l'emprunte au décodage ;
  témoin `ACVRAM_TETE_FP32_ENTREE=1`. Prédiction : −0,4 à −0,7 ms/pas (Sage −0,6).
- **(i) projections** : les int8 q/kv/o de l'attention MLA (qa_kv fusionné, q_b, o) portent
  `etroit` → `narrow_gemm` (tensor cores bf16, M ≤ 16) au lieu de `int8_gemv<4,12>` (31 µs par
  lancement pour ~3 Mo : calcul, pas octets). Portée limitée à ces tenseurs : le réglage global
  `ACVRAM_NARROW_GEMM` et le verdict Coder 0.6.6 (narrow OFF) ne bougent pas. Témoin
  `ACVRAM_NARROW_MLA=0` ; bras `ACVRAM_NARROW_ROWS=16` (1aj-2) à essayer. Même arithmétique à
  l'ordre des sommes près ; arbitre ≥ 80/84 à repasser. Prédiction : −1,5 à −2,5 ms/pas (Sage
  −2,0 ; réfuté < −1,0 → borné par les octets).
- Scellé Sage après (i)+(ii) : pas b=12 ≤ 19,0 ms (réfuté > 20,0), depuis 21,3.
