# La seconde perdue au premier préremplissage — protocole écrit avant de mesurer

Écrit le 8 septembre 2026, après que deux hypothèses ont été proposées puis
réfutées sans qu'aucune n'ait été formulée de façon falsifiable. Celui-ci l'est.

## Le fait

Le premier préremplissage coûte environ une seconde de plus que les suivants.
Il est **présent avec et sans graphes CUDA**, et il est **plus lourd sans**
(temps de première réponse 1000 ms contre 181 ms). Il subsiste après activation
du mode persistant, qui n'en retire qu'environ 22 %.

Deux coûts distincts coexistent et ont été confondus : un surcoût **avant** la
génération, mesuré par le temps de première réponse, et un surcoût **pendant**,
mesuré par le débit — que le banc calcule entre le premier et le dernier jeton,
donc **hors temps de première réponse**. Chaque explication proposée jusqu'ici
rendait compte de l'un et se présentait comme rendant compte du tout.

Ce document ne traite que du premier : la seconde perdue **avant** le premier
jeton.

## Ce qui est déjà éliminé, et par quoi

| candidat | éliminé par |
|---|---|
| compilation CUDA (PTX → SASS) | `~/.nv/ComputeCache` : 680 Mo, **rien écrit depuis le 24 août** |
| compilation Triton | `~/.triton/cache` : dernier fichier à **13h07**, séries à 18h25-19h |
| capture des graphes | journal : « godets capturés **d'avance** » ; et le coût est **plus lourd sans graphes** |
| chargement du processus | les cinq passages sont des requêtes **au même serveur déjà démarré** |

## Les quatre candidats restants

1. **Allocation du cache de clés-valeurs.** L'arène est réservée à la première
   requête, pas au chargement.
2. **Mise en service des poids exilés.** Premier transfert hôte → carte des
   couches placées en mémoire hôte.
3. **Premier passage sur les experts.** Un MoE ne touche un expert qu'une fois
   qu'un jeton le route ; le premier préremplissage en découvre beaucoup.
4. **Chargement différé des modules CUDA.** Le pilote ne charge le code d'un
   noyau qu'au premier appel de ce noyau (`CUDA_MODULE_LOADING=LAZY`, défaut
   depuis CUDA 11.7).

## Les mesures, et ce que chacune élimine

Chacune isole **un** candidat en le rendant impossible ou en le déplaçant, sans
toucher aux trois autres. À exécuter dans cet ordre : la première est gratuite,
les suivantes coûtent une exécution chacune.

### M1 — `CUDA_MODULE_LOADING=EAGER` (candidat 4)

Une variable d'environnement, aucune modification de code. Force le pilote à
charger tous les modules au démarrage du contexte au lieu du premier appel.

- **la seconde disparaît du premier préremplissage et réapparaît au démarrage du
  serveur** → c'est le chargement différé des modules. Rien à corriger dans le
  moteur : le coût est déplacé hors du chemin de service, ce qui est exactement
  ce qu'on veut. Le démarrage du serveur s'allonge d'autant.
- **rien ne bouge** → candidat 4 éliminé.

### M2 — deuxième requête après une première triviale (candidats 1 et 3)

Envoyer d'abord une requête d'**un seul jeton** au même serveur, puis la vraie.
La requête triviale alloue le cache de clés-valeurs et route quelques experts,
mais n'en découvre presque aucun.

- **la seconde disparaît entièrement** → le coût est l'**allocation du cache**
  (candidat 1), puisqu'un seul jeton suffit à la payer ;
- **elle diminue d'une petite fraction seulement** → l'allocation n'était qu'une
  part ; le reste dépend du **nombre d'experts touchés**, donc candidat 3 ;
- **rien ne bouge** → candidats 1 et 3 tous deux éliminés, ce qui ne laisse que
  le 2.

### M3 — même modèle, `ACVRAM_EXIL_COUCHES` à 0 puis à 12 (candidat 2)

Le test à une seule variable déjà prévu pour la question du transport, lu ici
sur le **temps de première réponse** et non sur la perplexité.

- **le temps de première réponse croît avec le nombre de couches exilées** → la
  mise en service des poids exilés est le coût, et il est proportionnel — donc
  chiffrable par couche ;
- **il ne bouge pas** → candidat 2 éliminé.

### M4 — contrôle, à ne pas omettre

Refaire M1 sur un modèle **entièrement résident** et sur un modèle **exilé**. Si
la seconde perdue existe aussi sur un modèle résident, les candidats 2 et 3
perdent la première place quel que soit le reste : un modèle qui ne transfère
rien ne peut pas payer un transfert.

## Table de décision, écrite d'avance

| M1 | M2 | M3 | conclusion |
|---|---|---|---|
| disparaît | — | — | chargement différé des modules ; déplacer au démarrage, rien à corriger |
| rien | disparaît | — | allocation du cache de clés-valeurs ; la préallouer au chargement |
| rien | diminue un peu | — | découverte des experts ; un préchauffage routant tous les experts la supprime |
| rien | rien | croît avec l'exil | mise en service des poids exilés ; coût par couche, chiffrable |
| rien | rien | rien | **aucun des quatre.** Ne pas en inventer un cinquième après coup : reprendre par un profil (`nsys`) qui dira où passe la seconde, au lieu de la deviner |

## Ce que ce protocole refuse

Il refuse de requalifier un écart en « bruit acceptable » après l'avoir vu, et
il refuse la cinquième hypothèse inventée quand les quatre premières ont échoué.
La dernière ligne de la table est un **profil**, pas une conjecture : deux
hypothèses ont déjà été proposées aujourd'hui sur ce coût, toutes deux
plausibles, toutes deux fausses, et aucune n'avait été écrite de façon à pouvoir
être réfutée avant qu'on la mesure.

## Ce que le protocole ne dira pas

Rien du surcoût **pendant** la génération (−7,5 % avec graphes, −3,2 % sans),
qui est un second coût, mesuré par le débit et non par le temps de première
réponse. Il faudra un protocole distinct : ne pas transporter les conclusions de
celui-ci vers celui-là.
