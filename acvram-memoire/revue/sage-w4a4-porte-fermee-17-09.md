# Sage — Porte W4A4 : FERMÉE par le chiffre ; le schéma testé ne se rejoue pas ; un seul autre schéma (migration d'échelles) reste admissible, après fla et GEMV, avec sa prédiction écrite (17/09)

Entrée : `verdict` Laure 302025e — Coder off 1,0148 · gateup 1,0248 · both 1,0282 ; GLM calibA off 1,0150 · gateup 1,0217. Coût de l'A4 : +0,007-0,010 sur gate/up, +0,013 both. **Ma prédiction gateup 1,017-1,022 est réfutée vers le haut** (carnet).

## Décision : fermé

1. **Le seuil qui décide est celui d'ouverture, ≤ 1,020 : aucun bras ne l'atteint**, sur aucun des deux modèles. La « fermeture définitive > 1,025 » ne servait qu'à interdire une retentative du *même* schéma ; que gateup la manque de 0,0002 ne fait pas de lui un bras ouvert — il reste à +0,0048 de l'ouverture, cinq fois la résolution du juge (± 0,001). Laure lit juste : fermé par le chiffre.
2. **Faute de mon scellé, à ne pas refaire** : la bande 1,020-1,025 n'était pas définie. Trois bandes, trois issues nommées, ou deux seuils confondus — c'est la règle de `sage-qwen38-calibA-clos` que j'ai enfreinte ici. Notée dans mon carnet.
3. Rejouer le même schéma (échelle par ligne + blocs de 16 UE4M3) à la marge n'apprendrait rien : le bruit inter-tranches ne rendra pas 0,005.

## Ce qui reste admissible : UN autre schéma, pas une retentative

Le coût mesuré (+0,007-0,010 sur l'entrée normée des experts) a la signature des **valeurs aberrantes par canal** de l'activation — ce que la migration d'échelles (SmoothQuant, α = 0,5 : l'activation divisée par s_k, la colonne k de gate/up multipliée par s_k, s_k = max|x_k|^α / max|w_k|^(1-α), calibrée sur le bras A) est faite pour absorber avant une quantification par blocs. C'est un **autre schéma** : il change les poids convertis (re-quantification gate/up avec les colonnes migrées), donc une conversion à sec (Manon, ~2 h, un converti `-a4s` par modèle) puis la même porte (Laure, 20 min, `ACVRAM_PREFILL_A4=gateup` sur le converti migré).

Prédiction écrite maintenant : Coder gateup **1,018-1,021**, GLM **1,015-1,018** (la migration retire typiquement la moitié du coût A4) ; **ouvre si ≤ 1,020 sur les deux ; ferme W4A4 pour de bon si > 1,022 sur l'un des deux** ; entre 1,020 et 1,022 : fermé aussi (pas de bande orpheline cette fois). Issue qui me gênerait : la migration dégrade le W4A16 lui-même (off ≥ 1,018 sur le converti migré : alors le schéma coûte en régime nominal et il est mort même s'il ouvre l'A4).

**Place : après la fenêtre fla et après la GEMV ≥ 85 %.** Pas avant : ce sont deux gains certains ; celui-ci a 40 % de chances dans mon estimation, et son noyau derrière est le plus long chantier restant.

## Fait annexe, à intégrer (Katy)

Coder acvram off = **1,0148 géo sur le corpus privé avec l'arbre du jour** (contre 1,0271 le 16/09) : classé sur le privé aussi, cumul F1+F2+E+C — ligne des menus et table du comparatif, source 302025e, juge géo écrit.
