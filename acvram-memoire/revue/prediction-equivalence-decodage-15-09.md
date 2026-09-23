# Prédiction scellée — mode « décodage » de equivalence.py

Océane, 15/09/2026, avant implémentation (pas une mesure GPU — je
recâble une fonction pure à partir de chiffres déjà mesurés par Laure,
revue/verdict-narrow-voyants-15-09.md).

## Rappel de la règle (Sage §7.2, sage-reprise-15-09-b.md)

Un nombre d'ulp fixe ne juge pas au décodage (5 ulp / cos≥0,9999 échoue
739/768 sur le témoin lui-même). Nouveau critère : le test calcule son
témoin à CHAQUE exécution (A-graphes vs A-eager, mêmes noyaux, mêmes
jetons) et B/A doit le DOMINER :
- `med(delta_ulp, B/A) ≤ 1,2 × med(delta_ulp, témoin)`
- `p90(delta_ulp, B/A) ≤ 1,2 × p90(delta_ulp, témoin)`
- `max(delta_ulp, B/A) ≤ max(delta_ulp, témoin)`
- toute divergence top-1 hors ex-aequo (B/A) ≤ `max(delta_ulp, témoin)`
Garde sur l'instrument : témoin avec `max > 150 ulp` OU `cos < 0,998` →
les graphes eux-mêmes ont bougé, le test s'ARRÊTE (invalide), il ne dit
pas "faux".

Le critère par position à 5 ulp/cos≥0,9999 (equivalence.py existant)
reste valable UNIQUEMENT pour le prefill à 2 couches où il a été
calibré — documenté comme un second régime, pas remplacé.

## Prédiction

Avec les chiffres réels de Laure (narrow A/B = B/A sous test, témoin =
A-graphes/A-eager) : med 6,4 vs témoin 7,0 ; p90 12,7 vs témoin 13,4 ;
max 48,8 vs témoin 87,8 ; cos 0,99925 (B/A) vs témoin 0,99909 — le
témoin est VALIDE (max 87,8 < 150, cos 0,99909 ≥ 0,998) et B/A le
DOMINE sur les trois statistiques → `verdict_decodage` doit rendre
`ok=True, invalide=False`. La divergence isolée "s3@3" (14 ulp, hors
ex-aequo) doit passer aussi (14 ≤ max(témoin)=87,8).

**Seuil de réfutation** : si `verdict_decodage` sur ces chiffres exacts
rend `ok=False` ou `invalide=True`, l'implémentation ne correspond pas
à la règle de Sage — à corriger avant tout usage.
