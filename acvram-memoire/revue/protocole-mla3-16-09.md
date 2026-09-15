# Protocole — MLA commit 3 (29ec688, `mla_prep_batch`) : trois bras d'env sur le même commit

poste3, 16/09/2026, avant mesure. Ordre : chef. Arbre mesuré : **travail/poste3-qa2 @ 29ec688** (poste4),
`campagne-mla3-16-09.sh` (bras nommés), régime du duel, `-k48`, sorties `scratchpad/mla3-16-09/`. Bras :
**A** = `ACVRAM_MLA_UNE_PASSE=0 ACVRAM_MLA_PREP_NOYAU=0` (= témoin 3f69c82), **B** = PREP seul (UNE_PASSE=0),
**C** = tout (défaut). nsys + équivalence sur A/B/C ; PPL 3 tranches sur A et C (B = intermédiaire).
Tests d'abord : test_mla_une_passe, test_mla_decode_batch, test_mla_norme, test_collect_mla.

## Scellés
- poste4 : lancements 2 541 → **≈ 1 850** ; pas B 19,5-20,5 ms ; pas C **15-17 ms** ; bit-identique ou ≤ 1 ulp.
- Moi : lancements C **1 800-2 000** (≈ 15 → 1 par couche : −47 × 14 = −660) ; B **19,0-20,5** (élémentaires 1,35 → ~0,4,
  trou 1,19 → ~0,8) ; C **16,0-17,5** (17,62 − ≈ 1,3) ; réfuté si C > 17,6 (le noyau ne remplace pas ce qu'il dit) ;
  équivalence : top-1 identique, `torch.equal` **faux mais ≤ 1 ulp** (einsum + RoPE fusionnés en fp32 : l'ordre change ;
  si bit-identique, tant mieux) ; PPL C/A **1,000 ± 0,001** par tranche (22 664 jetons, défaut connu de l'instrument).
Durée ≈ 3 + 9 + 4 + 18 min, unité `mla3-poste3`, en file derrière `mla2-poste3`.
