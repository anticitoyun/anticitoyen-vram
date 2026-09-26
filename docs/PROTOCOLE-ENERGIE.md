# Protocole de mesure d'énergie par jeton

Rédigé le 8 septembre 2026, **avant toute mesure**, pour rendre comparables
les jetons par kilojoule de quatre moteurs sur deux cartes bridées, RTX 5090 à
400 W et RTX 3080 Ti à 275 W. Personne n'a publié de mesure dans ce régime :
les articles relevés le 7 septembre portent sur H100, H200 ou une 4060 Ti à
pleine puissance. Le banc a déjà les colonnes `W` et `j_kJ` ; ce qui manquait,
c'est la discipline qui rend leurs chiffres vrais.

Chaque règle ci-dessous vient d'une erreur commise et payée sur une autre
instrumentation, la veille.

## 1. L'instrument avant la mesure

Ne pas intégrer des échantillons de puissance. Lire le **compteur d'énergie**
de NVML, `nvmlDeviceGetTotalEnergyConsumption`, en millijoules, monotone
depuis le dernier chargement du pilote. La différence de deux lectures est une
énergie exacte sur la fenêtre, sans hypothèse sur le pas d'échantillonnage.

`nvidia-smi dmon` à une seconde sert au **profil**, pour voir si la fenêtre
est un plateau ou une pointe ; jamais au total.

Vérifier d'abord que le compteur avance sur ces cartes : deux lectures à dix
secondes d'intervalle. S'il ne bouge pas, il n'est pas supporté et tout le
reste tombe — le dire, ne pas se rabattre en silence sur l'intégration.

## 2. Ce que la fenêtre contient doit être prouvé, pas nommé

Une fenêtre appelée « rafale » contenait trois secondes de calcul et cent
dix-sept d'autre chose ; trois séances concordaient parce qu'elles mesuraient
toutes la même chose, qui n'était pas ce que leur nom annonçait.

Donc : marqueurs pris **dans le moteur**, premier jeton émis et dernier jeton
émis, jamais au lancement du client. Prefill et décodage mesurés séparément,
ce sont deux régimes de puissance. Le chargement du modèle exclu, ou compté à
part, jamais fondu dans le total.

Vérification qui coûte peu : le profil `dmon` doit montrer un plateau qui
commence et finit aux marqueurs. Sinon la fenêtre ne contient pas ce qu'on
croit.

## 3. La ligne de base se prend après, et à chaque fois

Une ligne de base mesurée une fois au début a fait attribuer au calcul une
décroissance qui était la dérive de cette ligne. Le repos d'une carte dérive
aussi : ventilateurs, température, horloges qui redescendent.

Donc : repos mesuré **immédiatement après** chaque exécution, sur la même
durée que la fenêtre. Deux chiffres rapportés, brut tout compris et net repos
retranché, en disant lequel on compare.

## 4. Les cartes bridées sont un régime, pas un détail

À chaque exécution, enregistrer : limite de puissance effective
(`power.limit`), horloges SM, température de départ **et** d'arrivée, et les
motifs de bridage (`nvidia-smi -q -d PERFORMANCE` : SW Power Cap, HW Thermal
Slowdown).

Une exécution qui a touché le plafond thermique n'est pas dans le même régime
qu'une autre qui a touché le plafond de puissance : elles ne se moyennent pas.
Une carte à 400 W qui reste bridée en permanence mesure sa limite, pas le
moteur — c'est un résultat, à condition de l'écrire.

## 5. Le premier passage est froid

Cache d'allocateur, noyaux compilés à la volée, graphes CUDA à capturer, carte
froide. Un passage de chauffe jeté, puis **trois exécutions retenues**, avec
une exigence d'entrée : température de départ dans une bande de ±3 °C entre
exécutions, sinon on attend.

## 6. Ce qui rend deux moteurs comparables

Même modèle, même quantification, même prompt, même **nombre** de jetons
produits — pas même durée —, même contexte, serveur relancé entre deux
moteurs, une seule carte occupée et rien d'autre dessus. Vérifier par
`nvidia-smi` qu'aucun autre processus n'y vit, y compris un serveur
d'embeddings resté en route.

Le chiffre à comparer est le **joule par jeton net**, avec sa dispersion sur
trois exécutions.

## 7. Ce qui invalide une mesure, écrit d'avance

* un autre processus sur la carte pendant la fenêtre ;
* une dispersion des trois exécutions supérieure à l'écart qu'on prétend
  mesurer — l'écart doit valoir au moins trois fois l'écart-type, en dessous
  on ne conclut pas et on ne « tend » pas ;
* un motif de bridage différent d'une exécution à l'autre ;
* un nombre de jetons produits différent de celui demandé, troncature ou arrêt
  anticipé ;
* un compteur d'énergie qui n'a pas avancé.

## 8. Ce que le protocole ne mesure pas, dit d'avance

L'énergie du reste de la machine : processeur, mémoire, alimentation. NVML ne
voit que la carte. Un moteur qui déporte du travail sur le processeur paraîtra
plus économe qu'il n'est — et acvram, avec ses perceptrons exilés, est
exactement dans ce cas. Si l'écart entre moteurs descend sous 10 %, cette part
cesse d'être négligeable et il faut un wattmètre à la prise. À dire plutôt
qu'à ignorer.
