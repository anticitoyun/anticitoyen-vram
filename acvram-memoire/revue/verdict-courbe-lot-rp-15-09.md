# Verdict — recourbe du seuil de lot MMA avec route+pack : le godet 8 passe, le godet 4 non

Laure, 15/09/2026, 18:33-18:55, une prise de carte (courbe-rp-laure). Ordre :
Sage § 3 (main 8dc6237). Protocole scellé :
[`protocole-courbe-lot-rp-15-09.md`](protocole-courbe-lot-rp-15-09.md)
(laure cca394e). Moteur : arbre main d880637 (v0.6.4 + route+pack b451ede par
défaut, `_MOE_ROUTE_PACK` relu True au JSON ; MIN_T 9), relevé au journal.

## En-tête (REGLES §3)

    instrument   energie.py, compteur NVML ; cartes [0] ; fenêtres 23-36 s
                 (rondes ctx 2048, invite 256), repos 30 s ; -pl 400, horloge
                 libre ; température 35-41 °C avant, 43-55 °C pendant
    bras         A = MMA=0 ; B = MMA=1 + MIN_T=1 (forcée) ; route+pack des
                 deux côtés (ne touche que B) ; constantes relues au JSON
    godet        sous graphes la garde lit le godet (2/4/4/8 pour b=2/3/4/6)
    carte        vide début/fin ; aucune alarme « processus »
    données      scratchpad/courbe-lot-rp-15-09/certifie-b{1,2,3,4,6}-moyen-20s-{A1,B1,A2,B2}.json

## Résultats (moyenne des deux passes, écart entre passes ≤ 0,003 ms)

    b   A pas ms  A J/j   A W   | B pas ms  B J/j   B W   | ms B/A   J B/A  | sans route+pack (16:34)
    1   4,298     1,449   337   | 5,043     1,454   288   | +17,3 %  +0,3 % | +56,2 %  +14,2 %
    2   6,165     1,169   379   | 6,727     1,051   313   |  +9,1 %  −10,1 %| +36,3 %   −3,4 %
    3   7,285     0,963   396   | 7,844     0,902   345   |  +7,7 %   −6,3 %| +26,9 %   −6,2 %
    4   7,846     0,783   399   | 8,124     0,740   364   |  +3,5 %   −5,5 %| +24,1 %   +2,2 %
    6   10,644    0,698   393   | 9,524     0,622   392   | −10,5 %  −10,9 %| +18,0 %   +2,7 %

Route+pack a ramené le coût fixe de B de ≈ 2,0-2,5 ms à **0,56-0,75 ms**
(b=1 : 0,75 ; b=2-4 : 0,56-0,28 ; b=6 : −1,12) — bien plus que la moitié
que je prédisais.

## Contre le critère et les prédictions

**Critère de Sage** (J(MMA) ≤ J(GEMV) ET ms ≤ 1,02×) : **b=6 le satisfait**
(−10,5 % ms, −10,9 % J) ; b=4 non (+3,5 % ms, J bon) ; b=1-3 non (ms).
**Sage** : b=1 +10/15 (mesuré +17,3 : réfuté de peu), b=2 +3/8 (+9,1 :
réfuté de peu), b=3 −2/+3 (+7,7 : réfuté), b=4 −5/−10 (+3,5 : réfuté),
b=6 −10/−15 (−10,5 : **tenu**). Son « réfuté si b=4 ≥ 1,02× → MIN_T reste
9 » se déclenche à la lettre. **Moi** : réfutée partout (+22/+15/+10/+8/+4
prédits) — route+pack retire plus qu'une moitié de lancements, et à b=6 la
MMA gagne franchement.

## Ce que le seuil doit être (le godet, pas le lot)

Sous graphes la garde compare le GODET : b=5-8 → godet 8 ; b=3-4 → godet
4. Le godet 8 **passe** (b=6 mesuré, −10,5 % / −10,9 % ; b=5, 7, 8 non
mesurés mais encadrés par b=4 +3,5 % et b=12 −6,4 %), le godet 4 échoue à
b=4. À la lettre de Sage (b=4 ≥ 1,02×) : MIN_T reste 9 — mais 9 laisse
les lots 5-8 en GEMV alors que le godet 8 gagne 10 % en ms et en J.
**Recommandation : MIN_T = 5** (godets 8, 12, 16 en MMA ; 2 et 4 en GEMV),
avec une cellule b=5 (2 min) si Sage veut le bord du godet mesuré et pas
encadré. À Laurine pour v0.6.6 si Sage scelle.

## Alarmes déclenchées

A(b=4) 7,846 et A(b=6) 10,644 contre 8,125 et 10,923 à 16:34 (−3,5 % et
−2,6 %, hors ±2 %) ; A(b=1-3) inchangés. Différence entre les deux
sessions : v0.6.4 (routeur bf16) — qui devrait aussi toucher b=1-3, et
ne le fait pas ici ; non expliqué, relatif seul retenu (ABAB dans la même
session).
