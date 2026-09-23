# Verdict — MLA commit 2 (bb9fdc4, `mla_decode_1p`) : mla_* 6,45 → 2,32 ms, pas 21,49 → 17,62 ms, sorties bit-identiques ; bras narrow GLM : −0,65 ms seulement (les denses sont bornées par le lancement)

- **instrument** : identique à `verdict-mla1` (tests, nsys 50 pas b=12 ctx ≈ 300, ties 768 positions + 5 ulp, PPL décodage 3 tranches × 22 664 jetons) ; sorties `scratchpad/mla2-16-09/`
- **commit** : arbre mesuré **travail/laure-qa @ bb9fdc4** (laurine, main fusionné) ; témoin **`ACVRAM_MLA_UNE_PASSE=0` sur le même arbre** ; bras narrow `ACVRAM_NARROW_GEMM=1` (UNE_PASSE=1) ; scripts laure 8e683de
- **régime** : `-k48`, prefill W4A16, décodage MMA=1 MIN_T=5, MLA_BATCH=2, graphes actifs, lot 12 ; PREUVE lue par bras
- **scellé** : Sage/Jérôme : mla_* ≤ 3 ms (réfuté > 6), top-1 identique + cos ≥ 0,9999, PPL ± 0,004 ; Laurine : mla_* 1,5-2,5, pas 17-19 ; moi : mla_* 2,0-3,5, pas 17,5-19,5, ≈ 2 490 lancements, quelques refus 5 ulp ; narrow : denses 4,67 → 1,5-2,5 ms, pas −2 à −3
- **mesuré** : tests **15/15** ; **mla_* 2,32 ms** (`mla_1p_kernel<20,32,4>` 2,10 + combine 0,16 + écrit_latent 0,06 ; témoin 6,45) ; **pas 17,62 ms** (témoin 21,49) ; lancements **2 541** (inchangés) ; trou 1,19 ; équivalence **768/768, `torch.equal` vrai 12/12** ; PPL **B/A = 1,000000** × 3 tranches. Narrow : pas **16,83**, denses 5,85 → **5,21** (`narrow_gemm<32>` 3,75 + `<128>` 0,28 pour 387 lancements, à la place d'`int8_gemv<4,12>` 4,68)
- **verdict** : commit 2 **TENU** sur tout (mla_* ≤ 3, équivalence au-delà du scellé : bit-identique, PPL identique) ; Laurine tenue, moi tenue sauf les lancements (la passe unique n'en ôte aucun : 2 541 = élémentaires 2 000 + normes 406 + GEMV 388 + MoE 229 − …). Narrow sur GLM : **−0,65 ms (−3,7 % du pas), pas le −16 % de Coder** — ma prédiction réfutée, la cause est dite en § 1

## 1. Où en est le pas (ms/pas, même arbre, même instrument)
```
poste          témoin   commit 2   +narrow   lecture
mla             6,45     2,32      2,32     une passe : lit le latent une fois ; 2,1 ms pour 12 × 47 × (ctx ≈ 300) — restera ∝ ctx
denses          5,85     5,86      5,21     387 GEMV int8 (q_a q_b kv_a kv_b o + denses) à ~12 µs pièce = 4,7 ms pour 2,0 Go : 430 Go/s. Narrow tient à 4,0 ms : le poste n'est PAS la bande passante, c'est 387 lancements de petites matrices (N = 576, 512, 3 072…) → FUSION des projections partageant l'entrée (q_a + kv_a, k_b…), pas un autre noyau GEMV
moe_gemm        4,88     5,15      5,06     inchangé (bruit 0,3)
elementwise     1,36     1,36      1,34     ~2 000 lancements RoPE / cat / résidu (commit 3 de Laurine)
trou            1,21     1,19      1,18
total          21,49    17,62     16,83     vLLM même lot 15,1
```
- Ce qui sépare encore de vLLM (2,5 ms avec narrow) tient dans deux postes : **denses 5,2 ms** (fusion des projections : 387 → ~150 lancements attendus ≈ −2 ms) et **élémentaires + trou 2,5 ms** (`mla_prep_batch`, 89bb888, en file). Le MoE (5 ms à ctx court) et l'attention (2,3) sont désormais au niveau attendu.
- Colonne narrow pour Sage : sur GLM `ACVRAM_NARROW_GEMM=1` vaut −3,7 % (Coder : −16 %) ; il ne coûte rien de le laisser ON en 0.6.7 (déjà tranché sur Coder), mais il ne remplace pas la fusion.

## 2. Ce qui n'a pas été mesuré ici
Le contexte : 17,6 ms à ctx ≈ 300 ; la remesure `certifie` (rondes 256→2048, 10 min) donnera le chiffre publiable après fusion de laurine dans main — attendu ≈ 17,6 + 6-8 (KV lu par `mla_1p`, désormais le seul poste ∝ ctx). PPL absolue GLM : instrument tronqué 4/12 (voir `verdict-mla1` § 2), ratios seuls.
