# Verdict — xreg sur le down seul (Laurine 465030a, `ACVRAM_GROUPED_XREG=down`) : ABAB Coder b=12, **nu 9,18 contre 9,72 ms/pas (× 0,944)** → **TENU** (seuil ≤ 0,97) ; J/jeton × 0,954 ; bit-exact (1 024 NLL identiques)

instrument : `scratchpad/xreg-down-18-09/situ-abab.sh` — ABAB A = XREG=0 (témoin, rpw=4 défaut), B = XREG=down ; chaque bras `certifie-b12-15-09.py` (bridé 400 W, J/jeton, ligne régime `GROUPED_XREG=down` dans les JSON B) puis `profil-pas-coder-17-09.py … b12` (ms/pas nu) ; puis `ppl-decode-kv-17-09.py` Coder 256+1024 down / 0
commit : b786a8c (laure = main 4b83a2f + laurine 465030a), 08:36-08:45, une fenêtre
régime : classé, graphes on, b=12, rpw=4
scellé (Sage, sage-gemv-experts-clos-18-09) : moyenne B (ms/pas nu) ≤ 0,97 × moyenne A → tenu ; prédictions : Laurine 0,955-0,97, moi ≈ 0,96
mesuré : nu A **9,685 / 9,753** ms (1 239 / 1 230 t/s) · B **9,165 / 9,193** ms (1 309 / 1 305 t/s) → **B/A = 0,944** ; J/jeton A 0,3508 / 0,3520 · B 0,3341 / 0,3363 → **0,954** ; bridé A 10,52 / 10,58 ms 1 141 / 1 134 t/s · B 10,03 / 10,08 ms 1 196 / 1 190 t/s ; paires ABAB concordantes à 0,7 % ; ppl-decode-kv down **5,5426** = témoin 5,5426, 1 024 NLL par jeton identiques
verdict : **TENU** — −5,6 % de ms/pas nu (−0,54 ms, plus que les −0,4 prédits : le down libéré profite aux voisins), −4,6 % de J/jeton, sortie identique au bit ; les deux prédictions sont dépassées en mieux (0,944 < 0,955) ; règle 9 remplie pour `XREG=down` en défaut

## Lecture
- Le témoin A de cette fenêtre (9,72 ms nu) est 2 % plus lent que le rpw=4 de l'ABAB RPW (9,51, arbre d2670cb) : dérive d'arbre ou de fenêtre, sans effet sur le rapport ABAB (les quatre bras sont dans la même fenêtre) ; la cellule absolue, si Sage en veut une, se remesure sur main après fusion.
- Cellule Coder b=12 sous `XREG=down` (Katy, après décision de défaut) : nu 1 307 t/s (9,18 ms) / bridé 1 193 t/s / 0,335 J/jeton — le seuil in situ de 1 300 nu annoncé pour le geste complet est atteint par le down seul.
