# ABBA levier 2, 2e passe de confirmation — TENU dans la fourchette, gain confirmé — 22/09 (Manon)

* instrument : `scratchpad/laurine-b12-21-09/chaine-sampler-abba.sh` (identique à la 1ère passe), demandé par la Maîtresse pour confirmer un gain hors fourchette initiale (+6,84 %)
* commit : main 92657459
* régime : A = graphe (défaut) ; B = graphe+épinglé
* scellé (Maîtresse, confirmation) : ratio B/A attendu dans [1,02 ; 1,08], réfuté si < 1,01
* mesuré : A = 1552,2/1586,4/1561,9/1523,3 (méd 1557,1) ; B = 1591,0/1606,8/1622,5/1607,5 (méd 1607,2) ; **ratio B/A = 1,0322 (+3,22 %)** ; horloge_med A=2565,0 / B=2561,0 MHz (écart 0,16 %, réaliste)
* verdict : **TENU, dans la fourchette de confirmation** [1,02;1,08]. Les deux passes convergent sur un gain réel et significatif (1ère passe +6,84 %, 2e passe +3,22 %, moyenne pondérée ≈ +5,0 %) — la variance inter-passe (bruit connu ~50 t/s sur ce banc, cf. B écarts 1591-1622) explique l'écart entre les deux sans remettre en cause le signe ni l'ordre de grandeur. Le levier 2 (rapatriement épinglé) apporte un gain de débit b=12 réel, plus fort que la prédiction initiale d'Océane (+2,0-2,4 %).
* durée : ~9 min (04:34:17-04:43:37)

## Suite
Chiffre à publier : gain moyen ≈ +5 % (fourchette 3,2-6,8 % selon la passe), sous réserve d'une décision Sage/Maîtresse sur la valeur à retenir pour la cellule. nsys+familles-noyaux ensuite (relancé avec nsys DANS la prise carte.sh).
