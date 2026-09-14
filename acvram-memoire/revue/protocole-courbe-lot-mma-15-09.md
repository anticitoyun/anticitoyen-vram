# Protocole — courbe du lot b=2/3/4/6, MMA=0 contre MMA=1 (bascule v0.6.2)

poste3, 15/09/2026, avant mesure. Ordre : poste7
[`poste7-mma-lot-15-09.md`](poste7-mma-lot-15-09.md), transmis par chef.
Attend v0.6.2 de poste4 (`ACVRAM_MOE_DECODE_MMA_MIN_T`) sur main et la
carte après poste2.

## Critère de poste7 (scellé)

MMA retenu au lot `t` dès que **J/jeton(MMA) ≤ J/jeton(GEMV)** ET
**ms(MMA) ≤ 1,02 × ms(GEMV)** ; défaut = le plus petit `t` qui satisfait
les deux. Prédictions de poste7 (ms, relatif B/A) : b=2 **+20/+30 %**, b=3
**+5/+15 %**, b=4 **−3/+5 %**, b=6 **−8/−14 %** ; J croise un cran plus
bas ; attendu : **défaut 4 en J, 6 en ms**. Réfuté si b=6 encore ≥ 1,02×
en ms (la glue n'est pas le seul coût fixe).

## Montage

Même script que b=1 (`certifie-b12-15-09.py`, argument B), ABAB par lot,
20-25 s par cellule, rondes ctx 2048, invite 256, repos 30 s, compteur
`energie.py`, une carte, -pl 400, horloge libre ; **température GPU dans
l'en-tête** (avant / min / max pendant la fenêtre — l'alarme 223 vs 233 du
15/09 ne s'explique pas sans elle). Bras B = `ACVRAM_MOE_DECODE_MMA=1
ACVRAM_MOE_DECODE_MMA_MIN_T=1` (la MMA forcée à tout lot : la bascule est
ce qu'on calibre, elle ne décide pas ici) ; bras A = `MMA=0`. Bras et
`MIN_T` relus dans le module et écrits au JSON. 16 cellules, ~20 min.

## Mes prédictions ajoutées (scellées)

Coût fixe de B lu par poste7 : 1,54 ms/pas. Pas A attendus (rondes ctx
2048, pas b=1 de 4,48 ms et b=12 de 17,37 ms — interpolation par les
experts touchés, pas linéaire) : b=2 ≈ 6,0, b=3 ≈ 7,3, b=4 ≈ 8,5,
b=6 ≈ 10,8 ms. B = A − gain GEMM + 1,54 : je prédis **ms +22 % (b=2),
+9 % (b=3), +1 % (b=4), −7 % (b=6)** — dans les fourchettes de poste7 sauf
b=6 où je suis au bord (−7 contre −8/−14). J : **b=2 +5 %, b=3 −2 %, b=4
−7 %, b=6 −11 %** (B baisse les W de 20-27 %). Défaut attendu : **4 en J,
4 ou 6 en ms** (b=4 à ±2 % de 1,02×, c'est la cellule qui tranche).
Réfuté (mon bord) si b=4 est > +5 % en ms ou si b=6 est ≥ 1,02×.
Alarme d'avance : si A(b=2) n'est pas entre 5,5 et 6,5 ms, mon
interpolation est fausse, je ne garde que le relatif.
