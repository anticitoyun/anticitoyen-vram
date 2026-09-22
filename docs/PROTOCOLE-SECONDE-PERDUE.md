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

### La mesure fabrique sa propre pression : garde de faisabilité, calculée avant

Le relevé ci-dessus vérifie l'état **d'avant**. Il ne protège pas du cas où c'est
la mesure elle-même qui rend la machine inutilisable — arrivé le soir même : le
témoin bf16 a repoussé **4,9 Go en mémoire d'échange** en quelques minutes, et a
été arrêté par l'utilisateur.

Le mécanisme n'est pas la taille du modèle : c'est que les poids exilés sont
**épinglés** (`StreamedWeight`, `engine/layers.py:44,66-67`), pour que la copie
vers la carte reste asynchrone. La mémoire épinglée est **verrouillée en RAM par
construction** : le noyau ne peut ni l'évincer ni l'écrire en mémoire d'échange.
Les 4,9 Go swappés n'étaient donc pas nos poids — c'était **tout ce qu'ils ont
chassé** : navigateur, bureau, sessions.

Conséquence pour la garde : l'épinglage ne consomme pas seulement de la RAM, il
**retire au noyau sa marge de manœuvre**, cache de page compris. D'où le remède
qui a fonctionné (`swapoff -a && swapon -a`) et celui qui n'aurait rien donné
(attendre).

Cette garde-là se calcule **sans rien lancer**, dès l'écriture du protocole :

    octets épinglés prévus  contre  RAM libre − ce que le système doit garder

Le manifeste donne le premier terme (1,125 Gio par MLP × couches exilées =
33,75 Gio à 30 couches), `free` le second. Ces deux chiffres suffisaient à
refuser le lancement avant la lecture du premier octet.

Une garde qui ne sait dire non qu'après avoir vu le mal est en retard d'une
mesure.

### Quel compteur, et lesquels mentent

Trois candidats, et deux d'entre eux ne répondent pas à la question posée.
Tranché **sans rien charger**, sur `/proc/meminfo` de cette machine :

    Unevictable:      960 664 kB
    Mlocked:              132 kB

- **`VmSwap`** (`/proc/<pid>/status`) mesure ce qui a *été chassé* : un effet
  indirect, qui dépend de la pression exercée au même moment par le reste de la
  machine. Sur une machine calme il reste à zéro **avec ou sans correctif** — un
  succès apparent obtenu sans rien mesurer.
- **`Mlocked`** ne compte que le `mlock()` classique. Il est ici aveugle à
  **99,99 %** de ce qui est déjà verrouillé. Pris comme contrôle, il resterait
  plat pendant que 33 Gio s'épinglent.
- **`Unevictable`** compte les pages non évincibles — mais **seulement celles
  que `mlock()` ou `SHM_LOCK` ont marquées comme telles.**

> **RECTIFIÉ le 8/09 à 21 h, par la mesure, contre ce que j'avais écrit ici.**
> J'avais conclu que `Unevictable` voyait l'épinglage « quel que soit le
> mécanisme, CUDA compris », et que « c'est lui qui décide ». **C'est faux.** Il
> est resté à **0,00 pendant que 46 Gio étaient épinglés** par le moteur.
>
> L'épinglage CUDA passe par `pin_user_pages()` (**FOLL_PIN**) : le noyau laisse
> ces pages sur leur liste d'origine et les saute à la réclamation **sans les
> marquer non-évincibles**. `Unevictable`, `Mlocked` et `VmLck` y sont donc
> aveugles **par construction**, tous les trois.
>
> **Le compteur qui décide est `nr_foll_pin_acquired − nr_foll_pin_released`**
> dans `/proc/vmstat` (× 4096 octets), trouvé par Manon. Validé par concordance :
> `shmem` 46,0 Gio = `foll_pin` net 46,0 Gio, deux chemins indépendants.
>
> **Comment je me suis trompée, parce que c'est la partie utile.** J'ai éprouvé
> le compteur que je **rejetais** — le relevé `Mlocked 132 kB` contre
> `Unevictable 960 664 kB` — et jamais celui que je **retenais**. Ce relevé
> prouvait que `Mlocked` était aveugle ; il ne prouvait rien sur ce que
> `Unevictable` voit de *notre* charge. J'avais moi-même posé le préalable
> « vérifier qu'il réagit du tout au chargement d'un modèle exilé », je l'ai fait
> inscrire dans le patch de Jérôme, **et je ne l'ai pas appliqué à ma propre
> proposition.**
>
> **Règle qui en sort** : celle qui propose un contrôle est la moins bien placée
> pour l'éprouver. Un contrôle proposé doit être testé par quelqu'un d'autre —
> non par défiance, mais parce qu'on teste ce dont on doute et qu'on ne doute pas
> de ce qu'on vient de proposer.

Relever **avant chargement, après chargement, après déchargement**. Le retour à
la valeur initiale au déchargement est ce qui prouve qu'on mesure bien son propre
effet : sans lui, une dérive du système passe pour le correctif.

C'est la même forme que les autres pièges du dossier : un indicateur **voisin**
de celui qu'on croit lire. La mémoire disponible mesure ce qui reste à donner ;
elle ne dit rien de ce que le système est en train de reprendre.

### Borner la mesure, et pas seulement la refuser

Les gardes ci-dessus disent « ne commence pas si ça ne tient pas ». Elles ne
disent rien quand elles se sont trompées — et le 8 septembre elles se sont
trompées : le témoin a passé les préconditions puis paralysé la machine.

**La contrainte posée par l'utilisateur est qu'un test ne bloque JAMAIS le PC.**
Elle ne se satisfait pas d'une précondition, seulement d'un mécanisme qui borne
pendant l'exécution. Le contrôleur mémoire est délégué à la session utilisateur
(`cgroup.controllers` : `cpu memory pids`), donc sans `sudo` :

    systemd-run --user --scope --unit=<nom> \
        -p MemoryMax=24G -p MemorySwapMax=0 -- <la mesure>

**`MemorySwapMax=0` n'est pas facultatif — éprouvé, et le résultat est le
contraire de l'intuition.** Avec `MemoryMax=200M` seul, un programme d'essai a
alloué **4 Go sans broncher** : le noyau tient la borne de RAM en poussant le
surplus **en mémoire d'échange**. Une borne mémoire naïve *fabrique* donc
exactement le phénomène qu'on veut éviter. Les deux paramètres ensemble : mort
immédiate, sans sortie.

La victime devient toujours la mesure, jamais le bureau. Une mesure tuée se
relance ; une session utilisateur perdue, non.

**Borne d'accident, pas rationnement.** Le plafond doit être haut : il existe
pour rendre l'accident impossible, pas pour économiser. Une borne serrée sur un
processus qui a besoin de davantage ne le rend pas économe, elle le tue à chaque
redémarrage, à la même étape. Et `MemoryHigh` est à proscrire ici : au-delà, le
noyau n'arrête pas, il **étrangle** — un processus qui ne meurt pas et ne répond
plus est plus difficile à diagnostiquer qu'une mort franche.

**Réserve non levée** : rien ne prouve encore que la mémoire **épinglée par
CUDA** soit comptée par le contrôleur. Si elle lui échappe, la borne ne protège
pas de notre cas précis. S'éprouve en chargeant un modèle exilé sous une borne
délibérément trop basse : s'il ne meurt pas, on le sait avant d'en avoir besoin.

### Ce que le premier kill sous borne a appris

La borne a fonctionné : le noyau a écrit `constraint=CONSTRAINT_MEMCG`,
`oom_memcg=…temoin-moe`. **La machine a survécu, c'est le test qui est mort** —
ce qui est exactement le contrat. Décomposition du kill, `MemoryMax` à 45 Gio :

    anon-rss    0,57 Gio
    file-rss   26,86 Gio   pages fichier
    shmem-rss  44,11 Gio   mémoire partagée
    TOTAL      71,54 Gio

**`file-rss` établit directement que la source des poids est un mmap** — ces
pages ne peuvent être que des pages fichier. Le raisonnement par capacité est
corroboré par un relevé du noyau.

**Trois pièges dans la lecture de ce rapport, tous rencontrés :**

**`shmem-rss` n'est pas « verrouillé ».** Le cgroup sépare explicitement les deux
champs dans `memory.stat` (`shmem` et `unevictable`). Conclure de l'un sur
l'autre, c'est reprendre le compteur voisin. La question « la mémoire épinglée
par CUDA est-elle comptée par le contrôleur » **reste ouverte** ; seul
`unevictable` du scope y répondra.

**`MemorySwapMax=0` rend le shmem irréductible.** Le shmem n'a pas de fichier où
retomber : il n'est récupérable **que** par le swap. L'interdire rend donc tout
le shmem impossible à libérer, et le kill s'explique entièrement **sans** invoquer
d'épinglage. Ce n'est pas une raison de retirer le paramètre — il empêche la
fabrication de swap, qui est le mal qu'on évite — mais il **change le sens du
plafond** et doit être écrit à côté de lui.

**Un total relevé à la mort est une borne inférieure, jamais un besoin.** Le
processus a été tué **en cours de chargement** : il n'avait pas fini. Le besoin
réel est ≥ 71,54 Gio, d'un écart inconnu. Choisir le plafond suivant sur ce
chiffre le ferait tuer de nouveau.

**À relever pendant la mesure, et pas seulement avant et après** : `memory.stat`
du scope, champs `anon`, `file`, `shmem`, `unevictable`. Quatre lignes qui
répondent aux trois pièges ci-dessus.

### Un chien de garde se vérifie par ses battements, pas par son armement

La garde de ce lancement a écrit `GARDE ARMEE` puis `GARDE LEVEE` **dans la même
seconde**, avant que le scope surveillé n'existe : `systemctl is-active` sur une
unité pas encore créée rend faux, et la boucle n'a jamais été entrée. **Aucun
battement n'a été écrit, et personne ne l'a remarqué** — la ligne d'armement
avait été vérifiée, pas les battements.

**Un contrôle qui n'a jamais tourné est indiscernable d'un contrôle qui n'a rien
trouvé.** C'est la forme générale du `Mlocked` resté plat : **l'absence de signal
lue comme un signal d'absence**. Tout garde-fou de ce dossier doit donc écrire
un battement daté, et son absence doit invalider la mesure au même titre qu'une
alarme.

### `memory.peak` : sur le scope du test, jamais sur une slice partagée

`memory.peak` existe sur ce noyau et donne le **pic historique** du cgroup —
monotone, il ne redescend jamais. Relevé à l'instant sur `user@1000.service` :

    96 015 753 216 octets = 89,4 Gio     (sur 93,98 Gio de RAM)

C'est la trace chiffrée de la saturation du soir, et c'est aussi le piège : lu
sur une slice partagée, ce compteur mêle toutes les sessions et toute l'histoire
depuis le démarrage. Il n'a de sens que **sur un scope créé pour la mesure**,
où son pic est celui de la mesure et de rien d'autre. Encore un indicateur qui
porte le bon nom au mauvais endroit.

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

## Étape zéro, ajoutée le 9/09 : le phénomène existe-t-il encore ?

**À exécuter AVANT M1, et le protocole ne se lance pas si elle échoue.**

Ce document a été écrit le 8 septembre au soir. Depuis, trois choses ont changé
dans les conditions de toutes les mesures qui l'ont nourri :

* **`acvram-serveur` n'épinglait pas la carte** — seul des trois lanceurs.
  `_replanifier` recrutait alors les deux cartes, et une mesure prise là est un
  attelage 5090 + 3080 Ti, pas une 5090 ;
* **le `sync` tokensave** montait à 20 Gio et évinçait le cache de pages pendant
  les mesures ;
* **`ACVRAM_PLAN_FIGE`** n'était pas posé, donc le plan dépendait de la machine
  au moment du chargement.

**Conséquence sur ce document même, et il faut la dire avant de s'en servir.**

Le fait qu'il enquête — *« le premier préremplissage coûte environ une seconde de
plus »*, 1000 ms contre 181 ms — est un **relevé de temps**, pris dans ces
conditions-là. **Il se peut que la seconde perdue soit un artefact de l'attelage**
et non un comportement du moteur.

Et une des quatre éliminations en dépend : la ligne « capture des graphes »
s'appuie pour moitié sur *« le coût est plus lourd sans graphes »*, qui est une
**comparaison de durées**. Les trois autres éliminations reposent sur des
horodatages de cache ou sur la structure du banc — elles survivent.

**Étape zéro : refaire le relevé du temps de première réponse**, avec et sans
graphes, sur une machine désormais épinglée (`cartes=0`), à plan figé
(`plan_fige=1`), sans spéculation (`speculation=none`) et sans `sync` tokensave
en vol.

* **la seconde est toujours là** → le protocole part, et M1 à M4 gardent leur
  sens ; la ligne « capture des graphes » est à revérifier avant de s'y fier ;
* **elle a disparu ou fondu** → **il n'y a plus rien à expliquer**, les quatre
  mesures sont sans objet, et le fait à consigner est que le phénomène était
  l'attelage. Quatre mesures économisées, et une explication qu'on aurait
  cherchée pour rien.

**Ne pas dépenser quatre mesures à expliquer un phénomène avant d'avoir vérifié
qu'il existe encore.** C'est la forme la plus coûteuse du transport hors
conditions : non pas transporter une conclusion, mais transporter la **question**.

## CLOS — 9/09 : l'étape zéro a répondu, et les quatre mesures sont sans objet

**Réponse obtenue sans rien lancer** : les cinq TTFT étaient déjà dans le TSV de
la campagne comparative.

    acvram    146, 75, 75, 75, 75 ms   passage 1 = 146   regime = 75,0   +71 ms
    llamacpp  110, 39, 39, 33, 33 ms   passage 1 = 110   regime = 36,0   +74 ms

**La seconde perdue vaut 71 ms.** De 1000 ms contre 181 le 8 au soir à 146 contre
75 sur machine assainie : **facteur 14**. Le phénomène qu'allaient expliquer M1 à
M4 était très majoritairement l'attelage 5090 + 3080 Ti et le `sync` tokensave
en vol.

**Et le fait qui clôt le dossier : le surcoût est symétrique en absolu.** +71 ms
chez nous, **+74 ms chez llama.cpp**. Deux moteurs sans code commun payent le même
prix au premier préremplissage.

### Pourquoi « sans objet » est presque juste, et ce qu'il faut écrire à la place

Le protocole enquêtait sur un surcoût **propre à acvram**. La symétrie montre
qu'il n'y en a pas. Mais dire « sans objet » suggérerait que le phénomène était
imaginaire, et ce serait faux :

* **candidats 2 et 3** (mise en service des poids exilés, premier passage sur les
  experts) : **exclus par la symétrie** — llama.cpp n'a ni exil ni experts ici, et
  paie le même surcoût ;
* **candidats 1 et 4** (allocation du cache KV, chargement différé des modules
  CUDA) : **n'ont jamais été propres à acvram**. Un surcoût partagé de ~72 ms est
  exactement ce qu'ils prédisent, avec la sélection d'algorithme au premier appel.

**Formulation juste : le phénomène est réel, partagé, et trop petit pour compter.**
72 ms payés une fois par démarrage de serveur ne méritent pas quatre mesures. Ce
n'est pas une erreur de diagnostic corrigée, c'est un ordre de grandeur qui a
changé de trois décimales une fois l'environnement assaini.

**Valeur conservée pour plus tard** : si quelqu'un revoit un jour un premier
préremplissage à 1000 ms, il saura que **72 ms est le plancher** et que tout le
reste est de l'environnement — pas du moteur.

### Absolu ou relatif : la même donnée, deux conclusions opposées

    surcout absolu    acvram +71 ms    llamacpp +74 ms    -> ils sont un peu pires
    surcout relatif   acvram +95 %     llamacpp +206 %    -> ils sont deux fois pires

Même mesure, verdicts contraires, parce que leur régime est deux fois plus rapide.
**La forme se choisit d'après la décision qu'elle sert**, et le choix se déclare :

* *l'utilisateur attend-il plus longtemps ?* → **absolu**, en millisecondes ;
* *quelle part de sa propre performance le moteur perd-il ?* → **relatif**.

Publier l'une sans dire qu'on a écarté l'autre est un choix silencieux — et nous
savons ce que valent ceux-là.

### Une réserve sur le régime de llama.cpp

Leurs cinq TTFT sont `110, 39, 39, 33, 33` : **la série descend encore au
cinquième passage**, là où la nôtre est plate à 75 dès le second. Leur « régime »
à 36,0 est donc une moyenne sur une série **non stabilisée**. Sur une série plus
longue leur régime pourrait descendre encore, et l'écart de TTFT se creuser.
6 ms sur 33 peut n'être que la granularité du chronomètre — **non mesuré**, à ne
pas conclure dans un sens ni dans l'autre.
