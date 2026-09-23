# Verdict — après la tête fp32 (main 3ea31ae, qui fusionne aussi mla1-3) : arbitre graphes = GRAPHS_EAGER = eager (80 / 80 / 81 sur 84) ; pas b=12 GLM en régime juste **21,3 ms / 514 t/s / 0,774 J** ; l'erratum du duel s'inverse dans l'autre sens : ce n'est plus ×4,08 mais ×1,55

- **instrument** : arbitre prefill (84 points, prompt 128, k ∈ {1…63}) sur ties à `SLOTS = b` ; nsys 50 pas (rejeu) ; `certifie-b12` rondes ctx 256→2048, `energie.py`, ×2 ; sorties `scratchpad/post-correctif-16-09/`
- **commit** : **main 91a1320 = 3ea31ae** (tête fp32 + mla1-3 : `_MLA_BATCH` défaut 2, `mla_1p`, `mla_prep_batch`) ; laurine 3ea31ae identique côté code ; extension recompilée hors verrou
- **régime** : `-k48`, **W4A16 prefill et décodage** (Sage § 9), `ACVRAM_HYBRID_SLOTS=12`, graphes ; -pl 400, 2 782-2 895 MHz, 35-52 °C, 398 W (drapeau bridage puissance comme l'officiel Coder)
- **scellé** (Sage § 13 / Jérôme) : arbitre ≥ 80/84 GLM (cos ≥ 0,9999), Coder ≥ 26/28 ; pas b=12 ≤ +0,3 ms vs 17,10 ; remesure 56,1 ± 0,3 → ×4,08 confirmé
- **mesuré** : GLM 12 créneaux : **graphes 80/84 · GRAPHS_EAGER 80/84 · eager 81/84** (cos min ≥ 0,99995, |Δ| méd 1,1-1,2, k=2 11-12/12) ; **Coder b=4 graphes 27/28** (cos min 0,994, avant 0,872) ; laurine mla1-3 à 12 créneaux : 80/84 = main ; nsys pas b=12 (ctx ≈ 300) **20,51 ms**, 2 663 lancements, trou 1,24 ; **rondes : 21,27 / 21,32 ms · 514,6 / 513,5 t/s · 0,773 / 0,775 J · 398 W**
- **verdict** : arbitres **TENUS** (les trois modes coïncident, le correctif est le bon) ; « ≤ +0,3 vs 17,10 » **réfuté mais mal posé** : +3,4 ms = **+2,3 de W4A16 au décodage** (MoE `nvfp4_gemv_grouped_warp` 7,17 ms au lieu de la MMA 4,9 — c'est le régime de Sage § 9, pas le correctif) **+ 0,7 de tête fp32** (`int8_gemv<…,float,float>` 0,85 ms au lieu de ~0,18 — réductible : GEMV bf16 à sortie fp32) ; la « remesure 56,1 » n'existe plus : main n'est plus ce71723, le pas b=12 juste est **21,3 ms**

## 1. Le duel, colonne acvram réécrite (vLLM : prise A, inchangé)
```
                          acvram 16/09 matin (faux)   acvram main 3ea31ae (arbitre 80/84)   vLLM GadflyII NVFP4    rapport
b=12 t/s · ms/pas         195 · 56,1                  514 · 21,3                            796 · 15,1             ×1,55
J/jeton b=12              1,58                        0,774                                 0,445                  ×0,57
```
b=1 et b=4 : en cours (prise B, `duel-b-laure`, arbitre à SLOTS=b puis rondes) ; prefill inchangé par les commits MLA, remesuré pour la tête fp32.

## 2. Ce qui est acquis et ce qui reste
- Acquis : mla1-3 fusionnés et jugés à 12 créneaux avec la tête fp32 (80/84) ; le régime du duel est juste ; le plancher eager (66,1) est battu ×3 par le rejeu (21,3).
- Reste : la tête fp32 coûte 0,7 ms/pas (3 %) — Laurine peut la ramener à ~0,2 (GEMV bf16, accumulation et sortie fp32) ; le W4A16 au décodage coûte 2,3 ms (11 %) pour +1,4 % de PPL évités — un chiffre pour Sage, pas une décision de ma part ; b=1/4 et prise B ncu à suivre.
