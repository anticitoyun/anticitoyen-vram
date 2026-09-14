# Verdict — courbe du lot b=2/3/4/6, MMA=0 contre MMA=1 forcée (bascule v0.6.2)

poste3, 15/09/2026, 16:34-16:53, une prise de carte (courbe-lot-poste3, 1 109 s,
après 828 s d'attente derrière poste2). Ordre : poste7
[`poste7-mma-lot-15-09.md`](poste7-mma-lot-15-09.md). Protocole scellé :
[`protocole-courbe-lot-mma-15-09.md`](protocole-courbe-lot-mma-15-09.md)
(poste3 2e166d5). Code : main dd2f01f (v0.6.2, `_MOE_DECODE_MMA_MIN_T`
défaut 6, `model.py:1318`, garde `:1192`).

## En-tête de mesure (REGLES §3)

    instrument   energie.py, compteur NVML TotalEnergyConsumption
    cartes       [0] seule (exposée par carte.sh, champ `cartes` des JSON)
    fenêtre      23-36 s par cellule (rondes ctx 2048, invite 256), repos 30 s
    plafond      400 W ; horloge libre — A atteint le plafond dès b=3
                 (395-400 W, SM rabattue 2 650-2 860), B jamais (267-348 W,
                 SM 2 930-2 960)
    température  37-41 °C avant chaque cellule, 45-55 °C pendant (en-tête
                 de chaque JSON) — session homogène, ABAB dans la même heure
    bras         A = `ACVRAM_MOE_DECODE_MMA=0` ; B = `=1` + `MIN_T=1` (MMA
                 forcée à tout lot) ; constantes relues dans le module et
                 écrites au JSON (B : MMA True, MIN_T 1)
    godet        sous graphes la garde v0.6.2 lit le GODET (2/4/4/8 pour
                 b=2/3/4/6 — poste4), pas le lot réel : b=3 et b=4 partagent
                 le godet 4 ; la courbe mesure le lot réel, le seuil se lit
                 au godet
    carte        vide au début et à la fin ; aucune alarme « processus »
    données      scratchpad/courbe-lot-15-09/certifie-b{2,3,4,6}-moyen-20s-{A1,B1,A2,B2}.json

## Résultats (moyenne des deux passes, écart A1/A2 et B1/B2 ≤ 0,002 ms)

    b    A pas ms  A J/j    A W    A SM   | B pas ms  B J/j    B W    B SM   | ms B/A   J B/A
    1*   4,483     1,511    337    2964   | 7,005     1,726    246    2969   | +56,2 %  +14,2 %
    2    6,165     1,161    376    2944   | 8,403     1,122    267    2956   | +36,3 %   −3,4 %
    3    7,284     0,957    395    2864   | 9,244     0,897    291    2945   | +26,9 %   −6,2 %
    4    8,125     0,795    391    2769   | 10,084    0,813    322    2943   | +24,1 %   +2,2 %
    6    10,923    0,727    400    2648   | 12,890    0,747    348    2928   | +18,0 %   +2,7 %
    12*  17,367    0,620    391    2599   | 16,249    0,539    363    2916   |  −6,4 %  −13,2 %

    * b=1 : 054aafe (16:07, carte sortant de ncu) ; b=12 : c6377d5 (12:35).

## Contre le critère et les prédictions scellées

**Critère de poste7** (J(MMA) ≤ J(GEMV) ET ms ≤ 1,02×) : **aucun lot de 1 à 6
ne le satisfait** ; seul b=12 le satisfait. En ms, aucun lot ≤ 6 n'est
même sous 1,18×. **Réfuté** au sens de poste7 (b=6 ≥ 1,02×) : « la glue
n'est pas le seul coût fixe, le seuil attend route+pack ».

**Prédictions de poste7** (ms +20/30, +5/15, −3/+5, −8/−14) : tenue à b=2
(hors de peu, +36), réfutée à b=3, 4, 6. **Les miennes** (+22, +9, +1,
−7) : réfutées à b=3, 4, 6 ; mes pas A (6,0 / 7,3 / 8,5 / 10,8 prédits)
sont justes à 5 % — c'est B que ni elle ni moi n'avons su chiffrer.

**Ce que la courbe dit du coût fixe de B.** B − A = 2,52 / 2,24 / 1,96 /
1,96 / 1,97 ms à b=1/2/3/4/6, puis −1,12 à b=12. Ce n'est pas « 1,54 ms
qui s'amortit » : c'est ≈ 2,0 ms **constants de b=3 à b=6** (la
quantification A4 ×2 et la glue ne dépendent pas de t), et le gain des
GEMM n'apparaît qu'entre 6 et 12, quand A entre franchement au plafond
(SM 2 648 → 2 599) et que B, à 350-360 W, garde ses 2 930 MHz. Le
croisement en J à b=2-3 (B −3 à −6 %) vient de la puissance seule (B
−29 %), pas du travail ; il se défait à b=4-6 parce que A, bridé, devient
plus efficace par jeton.

## Conséquence pour le défaut v0.6.2

`MIN_T=6` sous graphes = godet 8 → **b=5 à 8 servis en MMA, soit −18 %
de débit et +2,7 % de J à b=6, mesurés**. Le seul lot où MMA gagne est 12
(godet 12/16). Jusqu'à route+pack : **MIN_T > 8** (la garde ne doit
prendre la MMA qu'aux godets ≥ 12), à poste4. Rien ici ne touche le
certifié b=12 (c6377d5).

## Trois états

    critère poste7 (b ≤ 6)   RÉFUTÉ à tout lot (ms 1,18-1,56×)
    défaut attendu 4/6     RÉFUTÉ — défaut mesuré : 12 (godet ≥ 12)
    en-tête température    présent (45-55 °C, session homogène)
