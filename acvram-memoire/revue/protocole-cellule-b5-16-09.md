# Protocole — cellule b=5 : MIN_T=5 (MMA au godet 8) contre MIN_T=9 (GEMV)

Laure, 16/09/2026, avant mesure. Ordre : Sage
[`sage-ordre-16-09.md`](sage-ordre-16-09.md), transmis par Jérôme. Arbre :
travail/laure figé à **b6e8d3a** (= main, v0.6.5 ; MIN_T défaut 9,
`model.py:1444` ; route+pack ; narrow selon le défaut de l'arbre au
lancement, relu au JSON). Carte après Manon (PPL narrow).

## Montage

Même script que la recourbe (`certifie-b12-15-09.py`, B=5) : rondes ctx
2048, invite 256, ≥ 20 s, repos 30 s, compteur `energie.py` corrigé
(b11b8a3), une carte, -pl 400, horloge libre, température en en-tête ;
ABAB : A = `ACVRAM_MOE_DECODE_MMA_MIN_T=9` (godet 8 < 9 → GEMV), B = `=5`
(godet 8 ≥ 5 → MMA) ; constante relue dans le module et écrite au JSON.
Sous graphes, b=5 tombe dans le godet 8 : cette cellule mesure le BORD du
godet que la recourbe avait encadré (b=4 : +3,5 % ms ; b=6 : −10,5 %).

## Critère (Sage)

MIN_T=5 adopté ssi **ms(B) ≤ ms(A) ET J(B) ≤ J(A) hors bruit**. Bruit
publié d'avance : écart entre passes du même bras (A1/A2, B1/B2) ; le
biais de durée est corrigé (fenêtres non arrondies), donc l'écart A/B en
ms est jugé à ± l'écart intra-bras.

## Ma prédiction (scellée)

À b=5, 5 jetons réels sur une tuile de 16 : coût fixe route+pack ≈ 0,3-0,6
ms contre un GEMV à 5 lignes. Interpolation de la recourbe (b=4 +3,5 %,
b=6 −10,5 %) : **ms B/A −2 à −5 %, J B/A −7 à −9 %** → MIN_T=5 adopté.
Réfuté si ms(B) > ms(A) (le bord du godet est à 6, MIN_T=6) ou si J(B)
> J(A). A attendu ≈ 9,1-9,5 ms (entre 7,85 à b=4 et 10,64 à b=6).
