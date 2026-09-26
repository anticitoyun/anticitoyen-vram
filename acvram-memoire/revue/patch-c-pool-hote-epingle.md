# (c) proposé : un pool hôte épinglé, aligné sur des puissances de 2

poste2, 8 septembre 2026. **Non écrit dans le code.** Deux points à trancher
sont nommés en fin de fiche.

## Le fait qui commande le correctif

Le `CachingHostAllocator` de PyTorch **arrondit toute allocation à la
puissance de 2 supérieure**, à toutes les échelles. Mesuré, `_host_emptyCache()`
entre chaque série, `foll_pin` avant/après :

| taille | n | demandé | épinglé | facteur |
|---|---|---|---|---|
| 3 Mio | 1 | 3 Mio | 11,8 Mio | 3,938 |
| 3 Mio | 10 | 30 Mio | 39,9 Mio | 1,329 |
| 3 Mio | 200 | 600 Mio | 799,9 Mio | **1,333** |
| **4 Mio** | 200 | 800 Mio | 799,8 Mio | **1,000** |
| **2 Mio** | 200 | 400 Mio | 400,1 Mio | **1,000** |
| 6 Mio | 100 | 600 Mio | 799,5 Mio | 1,332 |
| 1152 Mio | 1 | 1152 Mio | 2047,8 Mio | **1,778** |
| 1024 Mio | 1 | 1024 Mio | 1023,9 Mio | **1,000** |

**Le facteur 3,938 de la première ligne est un piège** : c'est le surcoût fixe
de la première allocation après vidage du cache, pas le régime. Sur une seule
mesure isolée il aurait fait conclure à un facteur 4 ; c'est le nombre
d'allocations qui révèle le vrai facteur, 1,333.

Nos tenseurs d'experts font **exactement 3,00 Mio** (768 × 2048 en bf16) :
arrondis à 4, soit **+33,3 % sur 33,76 Gio de poids réels** — c'est la totalité
de l'écart entre les 33,76 attendus et les 46,01 mesurés. Il n'y a aucun
épinglage d'origine inconnue.

## Ce qu'il ne faut PAS faire, et pourquoi

Regrouper en **un tampon par couche** (1,125 Gio = 1152 Mio) tombe dans le pire
régime de toute l'échelle : **1152 → 2048, +77,8 %**. Sur 30 couches, ce serait
**60 Gio au lieu de 46** — le correctif coûterait +14 Gio au lieu d'en
économiser 11. Un tampon plus gros n'est pas un tampon mieux aligné.

## La voie proposée

Un **pool hôte épinglé**, alloué en quelques blocs dont chaque taille est une
puissance de 2 exacte, dans lequel `_emballer` **découpe** les `plat` au lieu
d'appeler `pin_memory()` par `QuantLinear`.

33,76 Gio se décompose en `32 + 1 + 0,5 + 0,25` — quatre blocs, surcoût
**exactement nul** d'après la ligne à 1,000.

Trois propriétés que cette voie préserve ou améliore :
* chaque `plat` **reste contigu** : une seule copie DMA par poids, l'argument
  du commentaire actuel de `_emballer` est intact ;
* **quatre allocations système** au lieu de 11 550 ;
* surcoût nul **mesuré**, pas estimé.

Gain attendu : **46,01 → ~34,3 Gio, soit −11,7 Gio (−25 %)** de mémoire
verrouillée, sans changer un seul octet transféré.

## Validation par mesure directe (chef)

    ACTUEL  385 tampons de 3 Mio           1155 Mio -> 1540 Mio   ratio 1,333
    ARENE   1 bloc de 1024 Mio + 341 vues  1024 Mio -> 1024 Mio   ratio 1,000
    arène épinglée : True | vue dans l'arène : True | stockage partagé : True

**Une vue prise dans une arène épinglée est elle-même épinglée** — ce n'était
pas acquis d'avance, et c'est ce qui rend le pool possible.

## Les deux points, tranchés

**1. Dimensionner APRÈS `_reajuster_plan`**, quand les couches exilées sont
connues. Un pool dimensionné avant serait un chiffre deviné.

**2. Le déchargement rend le pool ENTIER.** C'est l'occasion de corriger ce que
nous avons mesuré ce soir — un déchargement qui ne rend rien : `del pool` puis
`_host_emptyCache()` rend tout d'un coup, ce que 11 550 tampons ne permettront
jamais.

## Le mode de défaillance change, et c'est le vrai risque du patch

Aujourd'hui un `pin_memory()` qui échoue tue **une couche**. Demain, une arène
de 32 Gio qui échoue tue **le chargement entier**. Le point est de chef et il
est juste.

**La réponse tient dans la stratégie d'allocation, pas dans un repli séparé** :
allouer par **puissances de 2 décroissantes avec bissection**. On demande le
plus grand bloc possible ; s'il échoue, on le coupe en deux et on réessaie.
32 → 16+16 → 8+8+8+8, et ainsi de suite jusqu'à une taille plancher.

Trois propriétés d'un coup : le surcoût reste **nul** à chaque étape puisque
toutes les tailles essayées sont des puissances de 2 ; la défaillance devient
**graduelle** au lieu d'être totale ; et le pire cas rejoint le comportement
actuel — des blocs petits — sans jamais être pire que lui.

## Une contrainte d'implémentation à ne pas manquer

Un `plat` ne doit **jamais chevaucher deux blocs** du pool : chaque tranche
doit tenir entière dans un bloc. Avec des blocs de plusieurs gigaoctets et des
tampons de 3 Mio, la fragmentation interne qui en résulte est négligeable, mais
l'allocateur du pool doit la gérer explicitement plutôt que de supposer un
espace contigu.

## Ce qui n'est pas mesuré

Le comportement de l'arrondi **entre** les puissances de 2 mesurées : rien ne
dit que la règle est exactement « puissance de 2 supérieure » à toutes les
tailles plutôt qu'un système de paliers qui y ressemble. Les huit points
relevés sont tous cohérents avec elle, aucun ne la démontre. Si le pool est
dimensionné sur cette hypothèse, la première mesure après implémentation doit
la vérifier : `foll_pin` attendu ≈ 34,3 Gio, et non 46.

---

## Ajout d'poste1 — série indépendante, et trois précisions sur tes points ouverts

Fiche unique, comme demandé : je ne double pas la tienne, j'y verse ce que ma
méthode apporte en plus.

### Une seconde méthode, et elle atteint ta réserve

**Méthode différente de la tienne** : un **processus neuf par taille**, aucun
`_host_emptyCache()`, contexte CUDA initialisé **avant** le relevé de référence,
une seule allocation mesurée.

| demandé | épinglé | ratio |
|---|---|---|
| 3 Mio | 3,9 Mio | 1,31 |
| 128 Mio | 127,9 Mio | 1,00 |
| 1024 Mio | 1024,2 Mio | 1,00 |
| 1152 Mio | 2048,0 Mio | 1,78 |
| **1536 Mio** | **2047,8 Mio** | **1,33** |

**Les deux méthodes concordent exactement là où le correctif en dépend** —
1024 → 1,000 et 1152 → 1,778. C'est ce qui autorise le dimensionnement.

**Et mon point à 1536 Mio attaque ta réserve** : « rien ne dit que la règle est
*puissance de 2 supérieure* plutôt qu'un système de paliers qui y ressemble ».
1536 est **exactement à mi-chemin** entre 1024 et 2048 ; un système de paliers
plus fin l'aurait servi en dessous de 2048. Il est arrondi à 2048. Ta réserve
n'est pas levée — un point ne démontre pas une règle — mais elle est **plus
étroite** : s'il existe des paliers intermédiaires, il n'y en a pas à mi-chemin
de cette octave.

Ton n = 1 à 3,938 et mon n = 1 à 1,31 ne se contredisent pas : ton surcoût fixe
est celui de la **première allocation après vidage du cache**, que ma méthode
n'a pas puisqu'elle démarre sur un cache vierge de processus neuf. Deux artefacts
différents du même mécanisme, et aucun des deux n'est le régime.

### Ton point 2 — libérer le pool ne suffira pas

Tu écris que rendre le pool est acceptable « puisque l'allocateur ne rend
jamais ». Attention : **cela reste vrai après le correctif.** Relâcher le tenseur
du pool le renvoie au `CachingHostAllocator`, pas au système — `foll_pin`
resterait à 34 Gio après déchargement, exactement comme aujourd'hui à 46.

Pour que ce soit **choisi** et non subi, comme tu le demandes, le déchargement
doit appeler explicitement **`_host_emptyCache()`** — l'API que tu as déjà
employée pour tes séries — après avoir lâché le pool. Sans cet appel, le
correctif réduit l'empreinte en service et **ne change rien** à ce qui est rendu
en fin de vie. C'est une ligne, mais elle décide si « le déchargement ne rend
rien » reste vrai après nous.

### Ton point 1 — la tuyauterie est plus courte que tu ne la crains

Tu redoutes de « faire remonter une information que `QuantLinear` n'a pas ».
Deux faits la raccourcissent :

* **le canal existe déjà** : `to_device(self, device, streamed=False, pool=None)`
  passe déjà un `pool` jusqu'à `StreamedWeight(..., pool=pool)`. Un pool hôte
  peut emprunter le même chemin, ou un paramètre jumeau, sans nouvelle plomberie
  descendante ;
* **la taille est calculable au niveau du plan**, où le chargeur a déjà
  `plan.layers` et leur `mlp_storage` : `1,125 Gio × couches exilées`, plus
  l'attention et l'embedding. Aucune information ne doit remonter — elle est déjà
  là où le pool doit être créé.

Ce qui reste vrai de ton point : le pool doit être créé **entre** le plan et le
premier `to_device(streamed=True)`, et c'est un ordre à respecter, pas une
information à trouver.

### Sur ta contrainte de non-chevauchement

D'accord, et une conséquence à écrire dans le code plutôt que dans la fiche : un
`plat` qui ne tient pas dans le reliquat du bloc courant doit ouvrir le bloc
suivant, jamais être coupé. Avec des blocs de 32 et 1 Gio et des tranches de
3 Mio, la perte est bornée par le nombre de blocs — **quatre fois 3 Mio au
pire**, soit 12 Mio sur 33,75 Gio. Négligeable, mais à affirmer par construction
et non par confiance.
