# Verdict — MLA 89bb888, bras A/B/C : `mla_prep_batch` ôte 706 lancements/pas mais coûte 0,95 ms pour en rendre 0,85 ; pas 21,58 → 17,17 ms avec les deux commits ; tout bit-identique

- **instrument** : identique à `verdict-mla1/2` (tests, nsys 50 pas b=12 ctx ≈ 300, ties 768 positions, PPL décodage 3 tranches × 22 664) ; sorties `scratchpad/mla3-16-09/`
- **commit** : arbre mesuré **travail/poste3-qa2 @ 89bb888** (poste4 : mla_decode_1p, mla_prep_batch, latent fp8 OFF) ; trois bras d'env sur le même arbre : **A** = UNE_PASSE=0 PREP=0 (témoin 3f69c82), **B** = PREP seul, **C** = défaut (tout) ; scripts poste3 e80a8ea ; le bras D (fp8) attend 3a1d2fd (unité `mla4-poste3` en cours)
- **régime** : `-k48`, prefill W4A16, décodage MMA=1 MIN_T=5, MLA_BATCH=2, graphes actifs, lot 12 ; PREUVE lue par bras ; tests 24/24 (fp8 désélectionné, misaligned address → 3a1d2fd)
- **scellé** : poste4 : lancements ≈ 1 850 ; B 19,5-20,5 ms ; C 15-17 ms ; ≤ 1 ulp. Moi : 1 800-2 000 ; B 19,0-20,5 ; C 16,0-17,5 ; `torch.equal` faux ≤ 1 ulp ; PPL C/A 1,000 ± 0,001
- **mesuré** : lancements **2 541 → 1 835** (A → B et C) ; pas **A 21,58 · B 21,43 · C 17,17 ms** ; trou 1,28 → 1,24 → 1,07 ; équivalence A-B, A-C, B-C : **768/768, bit-identique 12/12** ; PPL C/A **1,000000** × 3
- **verdict** : lancements **TENUS** (1 835 ≤ 2 500, poste4 ≈ 1 850 exacte) ; **B RÉFUTÉ des deux côtés** (21,43 : le prep seul ne gagne que 0,15 ms) ; C : poste4 réfutée de 0,17 ms (> 17), moi tenue (16,0-17,5) ; numérique identique au bit (mieux que ≤ 1 ulp). Le noyau `mla_prep_batch` est **neutre en temps** : il vaut par les lancements (graphes, trou), pas par les ms (§ 1)

## 1. Le compte du noyau de préparation (ms/pas, même arbre)
```
poste            A       B (prep)   C (prep + 1p)   lecture
mla             6,46     7,40       3,29            B : + mla_prep_batch_kernel 0,95 ms (47 lancements ≈ 20 µs) ; C : mla_1p 2,10 + prep 0,95 + combine 0,16
elementwise     1,36     0,59       0,59            −0,77 : les ~15 lancements/couche remplacés (einsum k_b, RoPE, cat, conversion)
normes          0,37     0,29       0,29            −0,08 (norme kv_a dans le noyau)
trou            1,28     1,24       1,07            −0,04 puis −0,17
denses          5,85     5,85       5,85            inchangé : 387 GEMV int8, le premier poste (34 % de C)
moe_gemm        4,89     4,89       4,92            inchangé
total          21,58    21,43      17,17            vLLM même lot : 15,1
```
- `mla_prep_batch` remplace 0,85 ms de petits noyaux par **0,95 ms d'un seul** : 20 µs par couche pour einsum k_b (20 têtes × 512 × 64 ?) + RoPE + norme sur 12 lignes, c'est un noyau **à un bloc par séquence** qui n'occupe pas la carte — le gain est ailleurs : −706 lancements (capture plus courte, trou −0,2 ms en C). À dire à poste4 : la grille (b = 12 blocs ?) sous-utilise 170 SM ; un bloc par (séquence, tête) rendrait sans doute les 0,95 → 0,2 ms — hypothèse, à lire dans le `.cu`, non mesurée.
- Ordre des gains sur le pas depuis 43,66 : batch (−22,2) ≫ une passe (−3,9) ≫ prep (−0,3 en C). Reste vs vLLM : **2,1 ms**, presque tout dans les denses (5,85 : fusion des projections à entrée commune, `verdict-mla2` § 1).

## 2. Ce qui est identique, et ce que cela vaut
Neuf comparaisons de logits (3 bras × 3 paires) et 6 PPL : **aucun bit ne bouge** entre A, B et C. Les trois noyaux (ecrit_latent, 1p, prep) reproduisent l'arithmétique fp32 de la référence à l'identique — il n'y a donc rien à trancher côté qualité pour les commits 1-3 hors fp8 ; seul le latent fp8 (bras D, 3a1d2fd) change des nombres, et c'est lui que la PPL ± 0,004 jugera.
