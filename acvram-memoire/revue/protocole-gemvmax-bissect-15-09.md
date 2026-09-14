# Protocole — marche 1 de 1aj (GEMV_MAX=8 à b=12) et bissection de la régression b=1

poste3, 15/09/2026, avant mesure. Ordre : poste7
[`poste7-1aj-decodage-15-09.md`](poste7-1aj-decodage-15-09.md) (main 0c90017),
transmis par chef. Carte après poste4 puis la reconversion de poste2.

## (1) Marche 1 — D à b=12, `ACVRAM_NVFP4_GEMV_MAX=8`

Bras A = D, GEMV_MAX=32 (défaut, `kernels/__init__.py:497`) ; bras B = D,
GEMV_MAX=8 → à M=12 les projections q/k/v/o NVFP4 quittent `nvfp4_gemv`
pour le chemin tensorcore/w4a8 (`:555-560`). ABAB 20 s, b=12, défaut
v0.6.3 (MMA au godet 12), une carte, température dans l'en-tête ; arbre =
worktree `travail/bissect/0c90017` (jamais l'arbre main), chemin
d'import écrit au JSON (`preuve.acvram`).

Scellé (poste7) : **17,92 → ≤ 16,4 ms** ; réfuté > 16,9. Ma prédiction :
16,3-16,8 ms — le w4a8 à M=12 sur des matrices étroites vaut à peu près
l'int8 par tranches (16,25 ms au témoin), pas mieux ; réfuté (mon bord)
si < 16,2 (le NVFP4 gagne aussi ses octets à M=12) ou > 17,0.

## (2) Bissection de b=1 : 233 t/s (14/09) → 223 (15/09)

Témoin int8-promu, `ACVRAM_MOE_DECODE_MMA=0`, b=1, ABAB 20 s, un
worktree par commit. good = **c652947** (main à 10:38 le 14/09, chiffre
233 de 089ec1f) ; bad = **0c90017** (main). Puis premier suspect de poste7 :
**97f2313 → 7f3f422** (routeur + biais en fp32, `model.py:1100-1117`).
Sources de noyaux identiques pour c652947/97f2313/7f3f422 (même hash),
différentes pour 0c90017 : aucune recompilation sous verrou.

Coupable si l'écart ABAB ≥ **0,13 ms** (poste7). Prédiction poste7 : 7f3f422
≥ 0,2 ms (réfuté < 0,1). La mienne : good/bad ≈ 0,19 ms (4,29 → 4,48) ;
7f3f422 en explique 0,10-0,20 — un cast fp32 + `F.linear` 128×2048 par
couche à M=1, 48 fois, ≈ 48 × 3 µs de lancements = 0,15 ms ; réfuté (mon
bord) si 7f3f422 < 0,08 (alors v0.6.1-3 ou 6-pré) ou si good/bad < 0,13
(la régression n'est pas dans le code : régime du 14/09 non reproduit).
Suite si 7f3f422 n'explique pas tout : b3811ca (0.6.1), dd2f01f (0.6.2),
c062988 (0.6.3), 18a00de (6-pré), carte.sh en dernier — worktrees et
recompilation à sec avant de reprendre la carte.
