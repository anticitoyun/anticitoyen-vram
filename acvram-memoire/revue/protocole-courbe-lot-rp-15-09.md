# Protocole — recourbe du seuil de lot MMA avec route+pack (b=1/2/3/4/6)

poste3, 15/09/2026, avant mesure. Ordre : poste7 § 3 (main 8dc6237), transmis
par chef. Moteur : main 8dc6237 (v0.6.4 + route+pack b451ede,
`ACVRAM_MOE_ROUTE_PACK=1` par défaut, `model.py:1348` ; `MIN_T` 9,
`:1346`). Carte après poste1 (5 min).

## Montage (identique à `protocole-courbe-lot-mma-15-09.md`, + b=1)

ABAB par lot, 20 cellules, rondes ctx 2048, invite 256, ≥ 20 s, repos 30 s,
compteur `energie.py`, une carte, -pl 400, horloge libre, température en
en-tête. Bras A = `ACVRAM_MOE_DECODE_MMA=0` ; bras B = `MMA=1 MIN_T=1`
(MMA forcée à tout lot) ; route+pack par défaut des deux côtés (il ne
touche que le chemin MMA) ; `_MOE_ROUTE_PACK`, `MIN_T`, `MMA` relus dans le
module et écrits au JSON. Sous graphes, le seuil retenu se lit au godet
(2/4/4/8 pour b=2/3/4/6).

## Critère et prédictions de poste7 (rescellées)

MMA retenu au lot dès que J(MMA) ≤ J(GEMV) ET ms ≤ 1,02× ; défaut = plus
petit lot qui satisfait les deux. ms B/A : b=1 **+10/+15 %**, b=2
**+3/+8 %**, b=3 **−2/+3 %**, b=4 **−5/−10 %**, b=6 **−10/−15 %** ;
attendu MIN_T 3-4 ; réfuté si b=4 ≥ 1,02× → MIN_T reste 9.

## Mes prédictions (scellées)

Sans route+pack le coût fixe de B était ≈ 2,0 ms/pas constant de b=3 à 6
(+2,5 à b=1). Route+pack retire ~2 200 lancements/pas à b=12 (poste4 :
3 677 → 1 517) ; à petit lot la part de lancements est la même par couche,
donc le coût fixe tombe d'environ la moitié, pas plus : **≈ 1,0-1,2 ms**.
Prédit ms B/A : b=1 **+22/+28 %**, b=2 **+15/+20 %**, b=3 **+10/+15 %**,
b=4 **+8/+12 %**, b=6 **+4/+8 %** ; J : b=2-3 −5/−8 %, b=4-6 −2/−5 %.
**Attendu : aucun lot ≤ 6 ne passe 1,02×, MIN_T reste 9** — je réfute
poste7 d'avance ; réfuté (mon bord) si b=4 ≤ 1,02× (alors la quantification
A4 ×2 ne pèse pas ce que je crois) ou si b=1 < +15 %. Alarme : A doit
retrouver 4,30 / 6,17 / 7,28 / 8,13 / 10,92 ms à ±2 % (courbe du 15/09
16:34 + v0.6.4), sinon régime différent, relatif seul.
