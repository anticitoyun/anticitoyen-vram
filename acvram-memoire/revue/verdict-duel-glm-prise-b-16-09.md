# Verdict — duel GLM prise B (main 3ea31ae : mla1-3 + tête fp32, régime W4A16 prefill et décodage, arbitré à SLOTS = b) : vLLM = ×1,43 / ×1,26 / ×1,55 notre débit à b=1 / 4 / 12 — scellé de Sage (0,8-0,9×) RÉFUTÉ de peu, plancher 0,7× tenu ; J/jeton : nous ×1,74 le leur à b=12

- **instrument** : arbitre prefill à `SLOTS = b` avant chaque lot (b=1 7/7, b=4 27/28, b=12 80/84 — `post-correctif`) ; `certifie-b12` rondes (ctx 2048 / invite 256 / ≥ 20 s, `energie.py` b11b8a3 + chrono hôte, écart ≤ 0,02 s) ×2 par lot ; prefill `prefill-glm-acvram` bras `w4a16` (7 rép.) ; vLLM : prise A (`duel-glm-16-09`, même ctx/invite, fenêtre 20 s) ; sorties `scratchpad/duel-b-16-09/` + `post-correctif-16-09/`
- **commit** : travail/laure **91a1320 = main 3ea31ae** ; protocoles 9998eca, 2cb4e22, f07ece8 ; vLLM 0.29.0
- **régime** : acvram `-k48`, **prefill W4A16 (`ACVRAM_MOE_MMA=0`), décodage W4A16 (`MOE_DECODE_MMA=0`, Sage § 9)**, `MLA_BATCH` 2 + une passe + prep, tête fp32, `HYBRID_SLOTS = b`, graphes, NOMINAL 0/47 ; vLLM GadflyII NVFP4 KV fp8 ; une carte, -pl 400, horloge libre, 38-52 °C
- **scellé** (Sage § 13) : prise B attendue **0,8-0,9 × vLLM** ; réfuté **< 0,7 ×** → trou hôte 5,3 ms d'abord. Moi (f07ece8, révisé) : nous 205-222 t/s à b=12 — **faux d'un facteur 2,3** (c'était le chiffre du régime faux)
- **mesuré** :
```
lot   acvram ms/pas      t/s     J/jeton    W     arbitre    vLLM t/s   J/jeton   nous / vLLM (t/s)   vLLM / nous (t/s)
b=1   9,335 / 9,308      107,3   2,263      243   7/7        153,7      1,97      0,70                ×1,43
b=4   11,980 / 11,982    333,9   1,028      343   27/28      421,1      0,77      0,79                ×1,26
b=12  21,272 / 21,317    514,0   0,774      398   80/84      796,3      0,445     0,65                ×1,55
prefill pp2048            4 422 j/s (W4A16, tête fp32 : 4 401 avant)   vLLM 26 732 j/s                          ×6,0
```
- **verdict** : **RÉFUTÉ de peu** — 0,65 à b=12 est sous 0,8 et juste sous le plancher 0,7 (b=4 : 0,79 ; b=1 : 0,70) ; à la lettre du scellé, le trou hôte (5,3 ms au 670cd63, 1,2 ms après mla1-3) passe devant. Le duel change de nature : **×1,55** au lieu de ×4,08 ; l'énergie reste ×1,74 (0,774 contre 0,445 J), le prefill ×6

## 1. Où sont les 21,3 ms (nsys 20,5 à ctx 300, `post-correctif/trous-lau.txt`)
`denses 13,6` (MoE W4A16 `nvfp4_gemv_grouped_warp` 7,2 + projections int8 4,4 + cuBLAS 1,0 + **tête fp32 0,85**) · `mla 3,5` (1p 2,2 + prep 1,0) · élémentaires 1,5 · trou 1,2 · 2 663 lancements. Trois leviers nommés, par taille : (a) **MoE W4A16 → MMA W4A4 : −2,3 ms** (c'est le régime, Sage § 9, +1,4 % de PPL à payer) ; (b) **projections int8 4,4 ms** : fusion des GEMV à entrée commune (387 lancements, ≈ 12 µs pièce) ; (c) tête fp32 0,85 → ~0,2 (GEMV bf16, sortie fp32). Le trou hôte (1,2 ms) n'est plus le premier poste.
## 2. Ce que la prise B ne mesure pas
vLLM : chiffres du matin, non arbitrés (sa PPL 1,056 contre notre 1,002 — le duel reste inégal en qualité) ; contexte : rondes 256→2048 des deux côtés ; prise B ncu (octets) non faite.
