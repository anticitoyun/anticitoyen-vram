# Prédiction scellée réfutée : graphes CUDA + chemin table RÉGRESSENT b=12

poste1, 14/09/2026. Suite de [[mesure-qui-tue-autopin-table-14-09]] après
avoir levé `graphs.py:224` pour le chemin table (commit `d44ea40`, 3
conditions prouvées par `test_repin_sous_graphe_cuda`). Prédiction scellée
de chef : « b=12 doit descendre sous 5× ». **Réfutée — dans le mauvais
sens.**

| régime (exil par expert, AUTOPIN) | b=1 (j/s) | b=12 (j/s) | facteur b=1 | facteur b=12 |
|---|---|---|---|---|
| graphes désactivés (mesure précédente) | 26,6 | 30,3 | 9,1 | 20,2 |
| graphes activés (chemin table, ce test) | 26,8 | **19,0-19,3** | 9,1 | **31,9** |

b=1 : neutre (26,6 → 26,8, dans le bruit). **b=12 : régression nette
(30,3 → 19,0 j/s, -37 %), facteur 20,2 → 31,9 — PIRE qu'avant de lever la
garde.** L'exil par couche (jamais concerné par ce chemin, toujours eager)
reste stable (16,8 / 40,9), confirmant que la régression vient bien du
changement, pas d'un artefact de mesure.

## Ce n'est pas un problème de recapture

Diagnostic (`captures`/`replays` de `GraphRunner`) sur le même régime,
b=12 seul, hors du banc à trois régimes :

    apres warm_graphs : captures=4 replays=4   (b=1 seulement, 4 godets)
    apres 200 pas     : captures=5 replays=203 (une seule forme b=12 en plus)

Une seule capture supplémentaire pour la forme b=12 (16,1,32,0), puis 199
rejeux directs. Le coût n'est donc PAS une recapture répétée — c'est le
REJEU lui-même qui coûte plus cher que l'exécution eager du même noyau.

## Hypothèse, NON VÉRIFIÉE

Le noyau table lit `table_qw[e]` (une adresse, donc une DÉPENDANCE DE
DONNÉES) avant de pouvoir lire le poids lui-même — double déréférencement,
contre un simple calcul d'offset (`e * M + row`) pour la pile contiguë. En
exécution EAGER, l'espacement naturel entre lancements (retour Python,
plusieurs flux d'activité qui se chevauchent) peut masquer cette latence de
dépendance. Un GRAPHE REJOUÉ compresse cet espacement au minimum — s'il
n'y avait auparavant un recouvrement caché, le rejeu l'exposerait
intégralement. Non vérifié par profilage (`ncu`) : hypothèse, pas mesure.

## Conséquence

**Le chemin table + graphes n'est PAS un gain net** pour le régime qui
compte le plus en service (lot concurrent, b=12) : léger mieux à b=1, net
pire à b=12. Je ne défais PAS le commit (les trois conditions de sécurité
restent vraies et testées — la garde levée elle-même n'est pas fausse),
mais je ne le recommande pas activé par défaut sans en comprendre la
cause. À trancher : profiler avec `ncu` avant de statuer, ou revenir à
« graphes désactivés pour le chemin table » en attendant.
