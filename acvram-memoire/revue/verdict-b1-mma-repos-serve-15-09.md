# Verdict — b=1 sous MMA=1 (défaut du produit) et repos sous `acvram serve`

poste3, 15/09/2026, 16:07-16:14, une prise de carte (b1-repos-poste3, 390 s,
après 2 008 s d'attente derrière `ncu-B.sh`). Ordre : poste7
[`poste7-reprise-15-09-b.md`](poste7-reprise-15-09-b.md). Protocole scellé :
[`protocole-b1-mma-repos-serve-15-09.md`](protocole-b1-mma-repos-serve-15-09.md)
(poste3 5b0aa7b). Code : main 02a33fa (`ACVRAM_MOE_DECODE_MMA` défaut 1,
`model.py:1294`).

## En-tête de mesure (REGLES §3)

    instrument   energie.py, compteur NVML TotalEnergyConsumption
    cartes       [0] seule (exposée par carte.sh, champ `cartes` des JSON)
    fenêtre      24-25 s par cellule (rondes ctx 2048, invite 256, b=1),
                 repos 30 s avant chaque cellule
    plafond      400 W relevé dans chaque processus ; horloge libre
                 (2 964-2 969 MHz moyens, jamais bridée : b=1 n'atteint pas
                 le plafond)
    mode         moyen ; bras PROUVÉ dans le processus (`model._MOE_DECODE_MMA`
                 relu : A False, B True, bt 16)
    carte        vide au début et à la fin ; aucune alarme « processus »
    données      scratchpad/ties-moe-15-09/certifie-b1-moyen-20s-{A1,B1,A2,B2}.json,
                 scratchpad/repos-serve-15-09/{repos.csv,energie-60-90.json,serve.log}

## 1. Chiffre officiel b=1 — MMA=1 RÉFUTÉ, largement

    bras  MMA   pas ms   t/s      J/jeton  W moy   jetons/fenêtre
    A1    0     4,483    223,04   1,5099   336,8   5 361 / 24,0 s
    B1    1     7,005    142,75   1,7241   246,1   3 574 / 25,0 s
    A2    0     4,483    223,04   1,5130   337,5   5 361 / 24,0 s
    B2    1     7,004    142,77   1,7270   246,6   3 574 / 25,0 s

    B/A : pas +56,2 %   débit −36,0 %   J/jeton +14,2 %   W −27 %

Reproductibilité ABAB : A à 0,000 ms près, B à 0,001 ms près.
Seuils de poste7 (ms −4 à −9 %, J −6 à −12 % ; réfuté si J ≥ 1,40 ou
ms > 1,02×) : **RÉFUTÉ des deux côtés** — J = 1,72 et ms = 1,56×. Ma
prédiction « ni gain ni perte (ms −1/+2 %) » est réfutée aussi, dans
l'autre sens : la tuile BT=16 pour un seul jeton ne coûte pas « rien »,
elle coûte +2,5 ms par pas (quantification A4 de l'entrée et de
l'activation, `nvfp4_quant_act` ×2, `model.py:1029/1033`, plus la GEMM
groupée à 1/16 de remplissage). Conséquence scellée par poste7 : **MMA=1 se
conditionne au lot** (b ≥ 4, geste `GardeSpeculation`) — à poste4. Tant
que le défaut reste « 1 », **le produit sert b=1 à 143 t/s au lieu de 223**.

**Alarme publiée d'avance, déclenchée** : A = 223,0 t/s / 1,51 J contre
233,0 / 1,42 le 14/09 (089ec1f, même script, même carte, même plafond) —
−4,3 % de débit, +6,5 % de J, reproductible sur A1/A2. Régime du jour
différent (carte sortant de 90 min de ncu ? état thermique non relevé —
à instrumenter : température dans l'en-tête). Le chiffre officiel b=1
MMA=0 est donc **223-233 t/s / 1,42-1,51 J** selon l'état thermique, pas
un point.

**Point 0 (égalité J avec llama.cpp à b=1)** : avec 1,51 (MMA=0) ou 1,72
(MMA=1) contre llama.cpp 1,17 (14/09, 3 moteurs) / 1,39 (chiffre de poste7),
**acvram perd le créneau b=1 énergie dans les deux bras** ; il ne revient
pas avec ce chiffre.

## 2. Repos sous `acvram serve` — CONFORME (≤ 25 W)

Serveur chargé (`--max-batch 4 --max-model-len 2048 --speculative none`,
défaut MMA=1, graphes actifs, 4 godets — `serve.log`), une requête de 64
jetons finie à 16:11:43, puis rien. Échantillons (t après la dernière
requête ; W ; P-state ; MHz) :

    +5  74,9 P1 2572   +10 74,6 P1 2572   +15 74,5 P1 2572
    +20 17,7 P8  225   +25 16,6 P8 225    +30 16,7 P8 225
    +61 15,9 P8  225   +66 16,0 P8 225    +92 17,2 P8 225
    compteur +90 → +120 s : 16,29 W moyen (489 J / 30,02 s), une carte

Seuil de poste7 ≤ 25 W à +60 s : **tenu** (15,9 W) ; ma prédiction (P8,
16-20 W, boucle `time.sleep(0.002)` sans commande CUDA,
`server/app.py:102-104`) tenue. La descente P1 → P8 se fait entre +15 et
+20 s, comme sans serveur (089ec1f). **Aucune attente active** : les bruts
b=1-4 n'ont pas à être révisés pour cette cause ; le « 68-76 W chaud » du
14/09 était une fenêtre de repos plus courte que la descente. Pour
l'en-tête de mesure : repos = 16 W (P8) après 20 s, 75 W (P1) avant.

## Trois états

    1. b=1 MMA=1   RÉFUTÉ  (pas +56 %, J +14 %) → conditionner au lot b ≥ 4
    2. repos serve CONFORME (15,9 W à +60 s, P8 dès +20 s)
    Point 0        perdu dans les deux bras (1,51 / 1,72 contre 1,17-1,39)
