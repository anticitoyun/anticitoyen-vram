# poste4 — chantier MLA (ordre poste7 `poste7-duel-verdict-16-09` § 3 et § 6), 16/09

## Commit 1 — chemin MLA du décodage batché au lot b (ce commit)
Constat (nsys de poste3, § 6.1) : 15 534 lancements/pas à b=12 (Coder : 1 517), dont 2 845
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

Prédiction scellée avant mesure (poste3, même nsys) : par couche MLA ≈ 30 lancements (2 GEMM,
2 normes, RoPE ≈ 10, cat ×3, einsum ×2 ≈ 4, écriture 1, attention 2, o_proj 1, conversions 2)
→ 46 × 30 ≈ 1 400 + le reste du pas (MoE, denses, ≈ 1 000) = **2 200-2 600 lancements/pas** ;
scellé poste7 ≤ 2 500 — si > 2 500, la marche suivante est nommée : RoPE + cat en un noyau
(≈ −10/couche). Pas b=12 : scellé ≤ 22 ms (réfuté > 26 → trou hôte, chantier suivant) ;
mon attendu 24-28 ms (les 5,6 ms de GEMV lot 1 et ~8 ms de cat/copies tombent, le trou de
5,3 ms reste). Équivalence : top-1 identique, cos ≥ 0,9999 contre `ACVRAM_MLA_BATCH=1` sur
16 positions × 2 couches ; PPL 3 tranches ± 0,004.

## Commit 2 — attention à une passe (prêt sur `poste4-mla-1p`, 0309a03)
`mla_decode_1p` : le cache latent lu une fois pour les H têtes (tuile de 32 lignes en
shared, scores fp32 des H têtes — pas de mma bf16 sur q : 2⁻⁸ sur des scores ~10 aurait coûté
le cos ≥ 0,9999 —, softmax en ligne, o_lat en registres, tranches de L recombinées) ;
`ACVRAM_MLA_UNE_PASSE=0` rejoue les deux noyaux. Scellé : mla_* 6,4 → ≤ 3 ms. Fusionné sur
poste4 après le verdict du commit 1.

## Commit 3 — latent fp8 par ligne (`poste7-avis-exterieur` § 6), après le 2.
