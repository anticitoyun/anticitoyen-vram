# Verdict — cellule b=5 : MMA au godet 8 égale le GEMV en ms et gagne 4 % en J → MIN_T=5

Laure, 16/09/2026, 06:16-06:22 (unité b5-laure), une prise de carte. Ordre :
Sage [`sage-ordre-16-09.md`](sage-ordre-16-09.md). Protocole scellé :
[`protocole-cellule-b5-16-09.md`](protocole-cellule-b5-16-09.md) (laure
ef0ea3e). Arbre : travail/laure figé à ef0ea3e (= main b6e8d3a, v0.6.5,
route+pack, narrow défaut 0), chemin d'import au JSON.

## En-tête (REGLES §3)

    instrument   energie.py corrigé (b11b8a3) + chrono hôte : écart 0,013-0,022 s
                 (garde < 0,05 s tenue) ; cartes [0] ; -pl 400 ; horloge libre
    fenêtre      34,2-34,4 s (2 rondes de 1 788 pas, ctx 2048, invite 256), repos 30 s
    température  37-39 °C avant, 49-53 pendant
    bras         A = MIN_T=9 (godet 8 → GEMV), B = MIN_T=5 (godet 8 → MMA) ;
                 `_MOE_DECODE_MMA_MIN_T` relu 9/5, `_MOE_ROUTE_PACK` True au JSON
    carte        vide début/fin ; données scratchpad/b5-16-09/b5-{A1,B1,A2,B2}.json

## Résultat

    bras         pas ms   t/s     J/jeton   W      SM MHz
    A1 (9)       9,585    521,6   0,7645    398,8  2734
    B1 (5)       9,579    522,0   0,7331    382,7  2938
    A2 (9)       9,627    519,4   0,7680    398,9  2721
    B2 (5)       9,575    522,2   0,7362    384,4  2936
    B/A (moy.)   −0,3 %   +0,3 %  −4,1 %    −3,7 %

Bruit intra-bras : A 0,042 ms (0,4 %), B 0,004 ms ; J : A 0,0035, B 0,0031.
**ms(B) ≤ ms(A)** à −0,3 % (dans le bruit : égalité) ; **J(B) < J(A)** de
4,1 % (hors bruit, 10× l'écart intra-bras). Critère de Sage tenu :
**MIN_T=5 adopté** (godets 8, 12, 16 en MMA ; 2 et 4 en GEMV). Ma prédiction
(ms −2/−5 %, J −7/−9 %) : le sens est tenu, l'ampleur non — le bord du
godet 8 est exactement à b=5 en temps (égalité), le gain est en énergie.
A attendu 9,1-9,5 : mesuré 9,58-9,63, hors de peu.

## Ce que ça dit

À b=5 la MMA lit autant de temps que le GEMV mais 15 W de moins (SM à
2 937 MHz au lieu de 2 727 : la GEMM sort du plafond 400 W où le GEMV
plafonne). Pour le godet 8, MMA = même débit, −4 % de J. À Laurine pour
v0.6.6 avec le défaut `ACVRAM_MOE_DECODE_MMA_MIN_T=5`.
