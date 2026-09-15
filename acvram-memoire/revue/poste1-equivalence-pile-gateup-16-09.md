# poste1 — équivalence pile/boucle GLM alpha-commun (16/09), RÉFUTÉ

Converti `GLM-4.7-Flash-srcbf16-nvfp4` (21:56:36). Deux couches (0 dense,
1 MoE), 16 positions, pile (nominal) contre boucle par expert (repli
forcé, `_stack_state="non"` sur chaque `MoEBlock`, mêmes poids chargés).
Critère `poste7-glm-equivalence §2` (`verdict_position`, 5 ulp).

**VERDICT : RÉFUTÉ.** Positions en échec : 8-15 (8 sur 16).

| position | delta | cos |
|---|---|---|
| 0-7 | 0,05-0,11 | 0,999978-1,000000 (bit-quasi-exact) |
| 8 | 1,8241 | 0,998601 |
| 9 | 1,5950 | 0,993729 |
| 10 | 1,1703 | 0,999734 |
| 11 | 0,9072 | 0,998746 |
| 12 | 1,0624 | 0,999336 |
| 13 | 1,3017 | 0,998736 |
| 14 | 1,5225 | 0,999548 |
| 15 | 1,2842 | 0,996724 |

**Coupure nette à la position 8** (préfixe de 9 jetons) : pas de
dégradation progressive, un saut. Pile à la lettre : ce n'est pas la
métrique de calibration qui est en cause (§2.1 aurait donné un écart
homogène, pas une bascule à un seuil précis) — piste : un seuil/bucket
dans le chemin `_forward_prefill_grouped`/MMA (godet, `_MOE_GEMM_MAX`,
taille de tuile) qui change de branche interne exactement à ce préfixe,
et où l'échelle alpha-commun n'est peut-être appliquée que sur UNE des
deux branches. Pas encore localisé précisément (fichier/ligne) — le
temps de carte alloué (10 min) est consommé par la mesure, pas
l'investigation ; carte rendue à poste4.

**Conséquence sur l'ordre de poste7 (§2)** : ce n'est PAS le cas
« conforme → passer au point 2 sans rien changer ». C'est un bogue
d'application à corriger avant toute chose (re-PPL scellé ≤ 1,010 après
correctif, réfuté si > 1,015).

**Prochaine étape (pas commencée)** : localiser pourquoi 8 est la
frontière — comparer `_forward_prefill_grouped` (godet/MMA) au chemin
`_forward_grouped`/gemv pour voir lequel des deux change de comportement
à 9 jetons, et si l'échelle alpha-commun n'est chargée que sur l'un des
deux.
