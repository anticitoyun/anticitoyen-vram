# Protocole — MLA commits 2-3 (89bb888 : `mla_decode_1p`, `mla_prep_batch`, latent fp8 OFF) : quatre bras d'env sur le même commit

Laure, 16/09/2026, avant mesure. Ordre : Jérôme. Arbre mesuré : **travail/laure-qa2 @ 89bb888** (laurine, remplace 29ec688 avant tout lancement),
`campagne-mla3-16-09.sh` (bras nommés), régime du duel, `-k48`, sorties `scratchpad/mla3-16-09/`. Bras :
**A** = `ACVRAM_MLA_UNE_PASSE=0 ACVRAM_MLA_PREP_NOYAU=0` (= témoin 3f69c82), **B** = PREP seul (UNE_PASSE=0),
**C** = tout (défaut), **D** = C + `ACVRAM_MLA_LATENT_FP8=1` (cache latent E4M3 par ligne, défaut OFF). nsys + équivalence
sur A/B/C/D (A-B, A-C, C-D) ; PPL 3 tranches sur A, C, D (B = intermédiaire).
Tests d'abord : test_mla_une_passe, test_mla_decode_batch, test_mla_norme, test_collect_mla.

## Scellés
- Laurine : lancements 2 541 → **≈ 1 850** ; pas B 19,5-20,5 ms ; pas C **15-17 ms** ; bit-identique ou ≤ 1 ulp.
- Moi : lancements C **1 800-2 000** (≈ 15 → 1 par couche : −47 × 14 = −660) ; B **19,0-20,5** (élémentaires 1,35 → ~0,4,
  trou 1,19 → ~0,8) ; C **16,0-17,5** (17,62 − ≈ 1,3) ; réfuté si C > 17,6 (le noyau ne remplace pas ce qu'il dit) ;
  équivalence : top-1 identique, `torch.equal` **faux mais ≤ 1 ulp** (einsum + RoPE fusionnés en fp32 : l'ordre change ;
  si bit-identique, tant mieux) ; PPL C/A **1,000 ± 0,001** par tranche (22 664 jetons, défaut connu de l'instrument).
- Bras D (latent fp8), Jérôme : à trancher par **PPL D/C dans 1 ± 0,004** (réfuté → bf16 gardé) et mla_* attendu −0,2 à −0,5 ms.
  Moi : mla_* 2,32 → **1,9-2,2 ms** (le latent lu est deux fois plus court ; le noyau n'est pas seulement borné
  par la lecture) ; PPL D/C **1,001-1,004** (E4M3 par ligne sur un latent de rang 512 : ~3 bits de mantisse, c'est
  la même famille de coût que le KV fp8 de vLLM, +0,8 % sur GadflyII entre bf16 et fp8 — mais dans l'autre sens
  là-bas) ; top-1 D/C ≥ 99 %, pas bit-identique ; réfuté (ma borne) si D/C > 1,004 → OFF reste.
Durée ≈ 3 + 12 + 5 + 27 min, unité `mla3-laure`, en file derrière `mla2-laure`.
