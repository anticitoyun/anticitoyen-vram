# Prédiction scellée — plancher W4A4/W4A16, GLM sans échelle (16/09)

Océane, avant mesure (Sage, sage-glm-w4a4-16-09.md).

Même script/critère que oceane-equivalence-pile-gateup-16-09.md, sur
`GLM-4.7-Flash-srcbf16-nvfp4-avant-alpha-experts-fix` (sans échelle
alpha-commun), MMA=1 (défaut). Delta(alpha-commun, mesuré) : max 1,8241,
médiane des positions 8-15 ≈ 1,332.

## Prédiction

Le plancher (arithmétique W4A4 vs W4A16 seule, sans aucune échelle) a un
delta du même ordre de grandeur que l'alpha-commun — les deux
arithmétiques divergent intrinsèquement, l'échelle n'ajoute rien de
notable.

**Seuil (Sage)** : delta(alpha-commun) ≤ 1,2 × plancher → pas de bogue,
c'est l'arithmétique W4A4 seule. > 2 × plancher → bogue dans
`_forward_grouped_mma`.

## MESURÉ : CONFORME, pas de bogue

Plancher (`-avant-alpha-experts-fix`, sans échelle), positions 8-15 :
deltas 1,0085-1,7917 (max 1,7917). Alpha-commun (mesuré plus tôt) :
1,8241 max. Ratio **1,02×** ≤ 1,2× → CONFORME. L'écart de la position 8
en avant vient de l'arithmétique W4A4 (activations E2M1) vs W4A16
(boucle), pas de l'échelle alpha-commun ni d'un bogue d'application —
confirme la lecture de Laurine (garde de lot `model.py:1340`, pas
l'échelle).
