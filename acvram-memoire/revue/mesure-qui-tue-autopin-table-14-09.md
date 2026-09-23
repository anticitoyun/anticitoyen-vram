# Mesure qui tue, remesure avec chemin table + profil AUTOPIN réel

Océane, 14/09/2026. Suite de [[mesure-qui-tue-experts-vs-couche-13-09]] après
les deux gestes demandés par Jérôme (13/09 soir) : (1) chemin groupé
table-aware (`_try_build_stacks`, `nvfp4_gemv_grouped_gateup_table` /
`nvfp4_gemv_grouped_table`), (2) profil AUTOPIN réel — trace M1 régénérée
(135 requêtes, 50194 jetons, 21 237 120 sélections, 442 440/couche en
moyenne, toutes au-dessus du seuil de confiance) → `.acvram_usage.json` du
modèle, `decider_residents` en mode AUTOPIN (plus « defaut »).

| régime            | b=1 (j/s) | b=12 (j/s) | facteur b=1 | facteur b=12 |
|-------------------|-----------|------------|-------------|--------------|
| résident complet  | 241,9     | 611,1      | 1           | 1            |
| exil par couche   | 16,4      | 39,9       | 14,7        | 15,3         |
| exil par expert (avant, defaut) | 13,5-13,7 | 30,1-30,6 | 17,3-19,8 | 19,8-19,9 |
| exil par expert (AUTOPIN + table) | **26,6** | 30,3 | **9,1** | 20,2 |

**b=1 : amélioration réelle et nette.** Facteur 17-20 → 9,1 (plus de 2×
mieux), et repasse enfin SOUS l'exil par couche (14,7) — ce que le chantier
visait. AUTOPIN (l'usage réel concentré, mesuré par M1) plus le chemin
groupé qui ne force plus la boucle lente pour les experts résidents
expliquent le gain.

**b=12 : quasiment inchangé (20,2), toujours pire que par couche.** Cible
<1,5 non atteinte dans les deux cas.

## Le plafond restant, identifié dans le code

`engine/graphs.py:224-231` : dès qu'UN SEUL `QuantLinear.streamed is not
None` existe dans le modèle, les graphes CUDA sont désactivés pour TOUT le
modèle — cette garde est ANTÉRIEURE au chemin table et protège contre un
danger réel mais spécifique : le chemin par expert/`ExpertPool` fait tourner
un pointeur à travers un jeu de tampons ROTATIFS — un graphe qui capture
« lire l'emplacement 3 » rejouerait un expert différent la fois suivante.

Le chemin TABLE n'a pas ce défaut par construction : le noyau relit
`table_qw[e]` à CHAQUE lancement (l'adresse vit dans un petit tenseur
device, pas gravée dans le graphe) — c'est exactement le contrat qui a
justifié la table d'adresses pour la MMA de Laurine (prefill). Rien ici ne
prouve qu'un graphe capturé sur le chemin table rejouerait faux — mais nier
que le risque a changé de nature sans le vérifier serait pareillement faux.
Deux conditions à établir avant de lever la garde pour ce cas précis,
NON vérifiées ici :

1. Une écriture de table (`_repin_echanger_reel`) est-elle garantie visible
   AVANT le prochain `replay()` qui la lit — quel flux, quelle
   synchronisation ? `_repin_pass` tourne en fin de `step()` (`runner.py`),
   après le replay du pas courant — reste à vérifier l'ordre exact contre
   le flux de calcul du graphe lui-même.
2. Le `_demote_expert`/`_promote_expert` change-t-il jamais la FORME
   (`shape`) d'un tenseur référencé par le graphe, pas seulement son
   contenu — un graphe capturé fige les formes, pas seulement les adresses.

C'est probablement le plus gros levier restant (le résident gagne le
graphe, aucun exilé ne l'a — cet écart à lui seul explique une bonne partie
du facteur 15-20 dans les DEUX régimes), mais c'est une garde de sécurité
mémoire, pas une optimisation de noyau : je ne la touche pas sans trancher
les deux points ci-dessus d'abord.

## Ce qui reste non expliqué

Pourquoi b=12 ne profite presque pas du chemin table alors que b=1 en
profite nettement — hypothèse non vérifiée : le blocage `pg[4] * 4 <= 48 *
1024` (mémoire partagée) ou un dispatch qui passe moins bien à l'échelle du
lot à G = 12 × top_k. Non creusé ici, faute de temps de carte.
