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

### Coïncidence de valeur n'est pas identité de mécanisme

Les allocations qui échouent pendant une évaluation font **20 971 520 octets**,
exactement la taille des godets de capture de graphes. La déduction « c'est donc
la capture » est fausse : `acvram eval` ne capture aucun graphe. 20 Mio est la
taille de bloc de l'allocateur, et les deux mécanismes la partagent sans se
ressembler. Ce raisonnement par ressemblance de chiffres a été démonté deux fois
dans la même soirée ; il figure ici pour ne pas l'être une troisième.

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

## Précondition : la mémoire disponible ne dit pas qu'une machine est calme

**À vérifier avant chaque mesure de ce protocole, et avant toute mesure de
temps en général.**

Le 8 septembre au soir, `free` annonçait **83 Go disponibles** au moment précis
où la machine était paralysée — souris et affichage compris. La charge était de
28,8 avec le processeur à **96 % d'inactivité** : elle n'attendait pas le calcul,
elle attendait le disque. `vmstat` donnait **17 800 blocs par seconde de
swap-in** : 8 Go chassés en mémoire d'échange une heure plus tôt par un
processus depuis longtemps mort, et qui remontaient.

Aucune de nos sessions n'allouait quoi que ce soit à cet instant. **La règle des
trois accords couvre ce que nous faisons ; elle ne couvre pas l'état laissé par
ce que nous avons fait une heure avant.**

Une seconde perdue se mesure en millisecondes. À ce débit de remontée, le
système en fabrique une sans que le moteur y soit pour rien — et le protocole
ci-dessous désignerait un candidat innocent.

    vmstat 1 3 | awk 'NR>3 {print $7}'   # colonne si, lignes 2 et 3 SEULEMENT
    # une valeur non nulle : NE PAS MESURER — attendre, ou dire pourquoi

**La première ligne de `vmstat` est un piège, et c'est le même piège que celui
que cette section dénonce.** Elle ne donne pas l'instantané : elle donne la
**moyenne depuis le démarrage de la machine**. Vérifié après le retour au calme,
swap vidé, charge à 1,6 :

    ligne 1 : si=10847     <- moyenne depuis le boot, contient les 8 Go d'il y a une heure
    ligne 2 : si=0         <- l'instantané
    ligne 3 : si=0

Un contrôle qui lit la ligne 1 refuserait de mesurer sur une machine parfaitement
calme, **et continuerait de refuser jusqu'au prochain redémarrage** — le compteur
cumulé ne redescend jamais. La garde deviendrait alors la première chose qu'on
désactive parce qu'elle « se trompe toujours », ce qui est la pire fin possible
pour une garde. Lire les lignes 2 et suivantes, jamais la première.

Relever `si` une seconde fois **après** la mesure : si le swap a repris pendant,
le temps mesuré est à écarter, pas à interpréter.

C'est la même forme que les autres pièges du dossier : un indicateur **voisin**
de celui qu'on croit lire. La mémoire disponible mesure ce qui reste à donner ;
elle ne dit rien de ce que le système est en train de reprendre.

## Les mesures, et ce que chacune élimine

Chacune isole **un** candidat en le rendant impossible ou en le déplaçant, sans
toucher aux trois autres. À exécuter dans cet ordre : la première est gratuite,
les suivantes coûtent une exécution chacune.

### M1 — `CUDA_MODULE_LOADING=EAGER` (candidat 4)

Une variable d'environnement, aucune modification de code. Force le pilote à
charger tous les modules au démarrage du contexte au lieu du premier appel.

**À lancer sur le chemin `serve`, pas sur `eval`.** `acvram eval` appelle
`load_model` directement (`evaluate.py:148`) et n'instancie aucun `Runner` : il
ne capture pas de graphes et ne sert pas de requêtes. Une M1 exécutée là
conclurait « sans effet » pour une raison qui n'a rien à voir avec le candidat
testé.

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
