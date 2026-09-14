# Équivalence CPU GLM-4.7-Flash, 2 couches — correctif routage, résultat intermédiaire

poste1, 14/09/2026. Bead urgent (chef, relais poste7) sur l'équivalence
rouge de poste2 (`outils/equivalence-glm-2couches.py`, revue/verdict-
equivalence-glm-2couches-14-09.md) : pire |Δlogit| 5,07 (seuil 0,05),
pire cosinus 0,952 (seuil 0,999), 8/16 positions sous le seuil.

## Piège de procédure trouvé en le rejouant : le sous-processus ne relit PAS le correctif

`etape_acvram()` lance `acvram convert` en sous-processus avec
`cwd=Path(__file__).resolve().parent.parent` — la racine du WORKTREE QUI
PORTE LE SCRIPT (`travail/poste2`), pas celui où le correctif vient d'être
écrit. `python -m acvram` ajoute le CWD en tête de `sys.path` : le
sous-processus importe `travail/poste2/acvram/`, une copie SÉPARÉE et non
corrigée, même si le venv est un lien éditable vers `travail/poste1`.
Vérifié directement : `load_model_spec()` appelé depuis mon worktree
rend `router_scoring="sigmoid"` (correct) ; la même conversion lancée
via le script de poste2, depuis son worktree, rend "softmax" dans le
manifeste — le correctif n'était simplement jamais vu.

**À nommer largement** : tout script qui lance `acvram` en sous-processus
avec le CWD de son PROPRE worktree peut silencieusement tester le code
d'un AUTRE worktree que celui qu'on croit — piège générique, pas
spécifique à ce bead.

## Correctif de routage (poste7, revue/poste7-refutation-glm-routage-14-09.md)

`config.py` : `router_scoring="sigmoid"` forcé pour `model_type ==
"glm4_moe_lite"` (sa config HF ne déclare pas `scoring_func`, et le
repli générique tombait sur "softmax", qui ignore le biais de
correction). Pas un heuristique "sigmoid si biais" : `ernie4_5_moe`
garde son softmax légitime. 4 tests (`tests/test_routage_glm4_moe_lite.py`),
cassent si le cas spécifique est retiré (vérifié manuellement).

## Résultat, correctif appliqué CORRECTEMENT (cwd forcé sur mon worktree)

```
position  0: Δ=1.87  cos=0.999548
position  1: Δ=1.56  cos=0.996235   <- reste mauvais
position  2: Δ=0.14  cos=0.999987
position  3: Δ=2.36  cos=0.994423   <- reste mauvais
position  4: Δ=0.15  cos=0.999985
position  5: Δ=5.11  cos=0.949580   <- reste TRES mauvais
position  6: Δ=0.24  cos=0.999997
position  7-15 : tous cos >= 0.999962, Δ <= 0.22

pire_delta=5.114 (seuil 0.05) pire_cos=0.949580 (seuil 0.999)
VERDICT=RÉFUTÉ (encore) -- mais 13/16 positions passent le seuil cosinus,
contre 8/16 avant. Le routage était la cause DOMINANTE, pas la seule.
```

Le routage sigmoid+biais explique la majorité des positions (8 corrigées
sur 8 mauvaises). Trois positions (1, 3, 5) restent nettement fausses,
position 5 très au-delà des deux autres — signature plus proche d'une
VRAIE divergence de sélection (top-k qui bascule sur un autre expert)
que d'un bruit numérique diffus, au vu de l'implémentation du routage
sigmoid déjà lue (`model.py:_route`, 1052-1083) qui suit fidèlement le
schéma HF (sélection sur scores+biais, poids gathered SANS biais,
renormalisation, échelle) — pas de défaut évident à la lecture.

## Suite (pas encore faite)

Contrôle de poste7 : comparer `topi` (indices d'experts sélectionnés,
couche 1 MoE) entre acvram et HF sur les positions 1, 3, 5 précisément —
si différents exactement là, c'est une bascule de sélection (bruit
bf16 sur la frontière top-4, ou biais mal aligné sur un sous-ensemble
d'experts) ; si identiques, regarder les poids/expert partagé, puis
l'attention (v_head_dim 256, jamais posé chez nous avant ce modèle).

Pas encore mesuré. Ce document sera complété avant de conclure — pas de
verdict final tant que la cause des 3 positions restantes n'est pas
nommée.
