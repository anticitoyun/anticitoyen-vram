# Protocole — MLA commit 2 (bb9fdc4, `mla_decode_1p` par défaut) + bras `ACVRAM_NARROW_GEMM=1` sur les denses GLM

Laure, 16/09/2026, avant mesure. Ordre : Jérôme (+ demande de Laurine pour le bras narrow). Arbre mesuré :
**travail/laure-qa @ bb9fdc4** (laurine, main fusionné), témoin `ACVRAM_MLA_UNE_PASSE=0` (ancien code, même arbre).
Même campagne que commit 1 (`campagne-mla1-16-09.sh`, `VAR=ACVRAM_MLA_UNE_PASSE VALS="1 0"`, `NSYS_NARROW=1`,
`TESTS=test_mla_une_passe + test_mla_decode_batch`), régime du duel, `-k48`, sorties `scratchpad/mla2-16-09/`.

## Scellés
- Sage/Jérôme : poste **mla_* ≤ 3 ms** (réfuté > 6) ; équivalence **top-1 identique ET cos ≥ 0,9999** contre
  UNE_PASSE=0 (pas bit-identique attendu : ordre des sommes fp32) ; **PPL B/A dans 1 ± 0,004** (3 tranches).
- Laurine : mla_* 6,4 → 1,5-2,5 ms ; pas 21,4 → 17-19 ms.
- Moi : mla_* **2,0-3,5 ms** (scores + reduce fusionnés lisent le latent une fois : 6,4 → moitié ou mieux ; réfuté
  > 4 → la passe unique ne recouvre pas la lecture) ; pas **17,5-19,5 ms** ; lancements 2 541 − 47 = **≈ 2 490**
  (un noyau au lieu de deux par couche) ; top-1 ≥ 99,5 % et cos ≥ 0,9999 sur 768 positions, critère 5 ulp
  d'Océane : quelques refus (≤ 10) attendus par l'ordre des sommes ; PPL B/A = 1,000 ± 0,001 par tranche.
- Bras narrow (verdict à part) : `int8_gemv<4,12>` 4,67 ms → **narrow int8 GEMM 1,5-2,5 ms** (Coder : −16 % du pas
  à b=12 par ce seul levier) ; pas total **−2 à −3 ms** ; réfuté si denses > 4 ms (le chemin narrow n'est pas
  pris sur GLM : K ou N non multiples de 64, à lire dans `kernels/__init__.py:681`) — preuve `NARROW` au PREUVE.
Durée ≈ 2 + 9 + 3 + 15 min, unité `mla2-laure`.
