# Verdict — poste4 976a090 (+54d5b28) : GEMM étroit sur les projections MLA −0,23 ms, tête GEMV ≤ 0,14 ms ; pas b=12 **21,06 / 21,15 ms** — scellé ≤ 19,0 RÉFUTÉ (> 20,0 aussi) ; arbitre 82/84

- **instrument** : tests (44/44) ; arbitre prefill à SLOTS=12 ; nsys 50 pas × 3 bras (A tout, B `ACVRAM_NARROW_MLA=0`, C `ACVRAM_TETE_FP32_ENTREE=1`), analyse par poste ; `certifie-b12` rondes ×2 ; sorties `scratchpad/narrow-mla-16-09/`
- **commit** : arbre mesuré **travail/poste3-qa @ 976a090** (poste4, extension recompilée hors verrou) ; référence main 3ea31ae : nsys 20,51, rondes 21,27 / 21,32
- **régime** : W4A16 prefill et décodage, `-k48`, SLOTS=12, graphes, tête fp32 ; -pl 400, 2 805-2 895 MHz, 34-51 °C, 396 W
- **scellé** (poste7 / chef) : pas b=12 ≤ 19,0 ms, réfuté > 20,0 ; arbitre ≥ 80/84 (l'ordre des sommes change avec le GEMM étroit)
- **mesuré** : arbitre A **82/84** (k=2 12/12, cos min 0,99999) ; nsys **A 20,11 · B 20,34 · C 20,25 ms** (2 662-2 663 lancements, trou 1,21-1,26) ; rondes A **21,056 / 21,153 ms · 519,8 / 517,5 t/s · 0,762 / 0,765 J**
- **verdict** : **RÉFUTÉ** — 21,1 ms, ni ≤ 19,0 ni ≤ 20,0 ; gain réel **−0,23 ms** (−1,1 %) sur les 21,3 de main. Arbitre tenu (82 > 80). Le levier « −2,0 ms attendus » n'a pas été atteint parce que le GEMM étroit ne change pas ce qui coûte (§ 1)

## 1. Attribution (nsys, ms/pas)
```
                        A (tout)    B (NARROW_MLA=0)   C (tête témoin)   lecture
denses                  13,27       13,45              13,46
  nvfp4_gemv_grouped     7,11        7,1                7,1              MoE W4A16 : le premier poste, intouché (régime poste7 § 9)
  narrow_gemm<32>        2,63        —                  2,6              remplace ≈ 2,8 ms d'int8_gemv (4,36 → 1,60) : −0,2, pas −2,0
  int8_gemv<4,12>        1,60        4,36               1,6              ce qui reste en GEMV (denses hors MLA + tête ?)
  gemmSN (cuBLAS)        1,02        1,02               1,02
mla                      3,45        3,47               3,45
total (nsys)            20,11       20,34              20,25
```
- Les projections MLA int8 (≈ 2,0 Go/pas) tournent à ~430 Go/s en GEMV et à ~700 Go/s en GEMM étroit : le tensor core n'y change que 0,2 ms parce que le coût est **par lancement** (387 lancements ≈ 7 µs pièce, dont ~5 de latence) — c'était le diagnostic du matin (`verdict-mla2` § 1) : il faut **moins de lancements**, pas des lancements plus rapides. La « fusion » que poste7 visait était celle-là (q_a+kv_a partagent l'entrée : 2 → 1 ; q_b et o restent séquentiels) ; poste4 a raison qu'ils ne sont pas tous fusionnables, mais 387 → ~250 lancements est atteignable, et le vrai plafond est le GEMV groupé du MoE (7,1 ms, W4A16), qui ne bougera qu'avec la MMA W4A4 (−2,3 ms, +1,4 % de PPL) ou un GEMV plus large.
- Tête : le témoin C ne montre pas de noyau `<float,float>` distinct — soit l'ancienne conversion n'est plus prise sous `TETE_FP32_ENTREE=1` dans ce commit, soit le coût est dans le même noyau : 0,14 ms d'écart, à confirmer par poste4 ; l'estimation « 0,85 → 0,2 » du matin venait d'un nom de noyau, pas d'un A/B.
## 2. Colonne du duel : 21,06 ms · 520 t/s · 0,762 J (vLLM ×1,53) — inchangée à 1 % près.
