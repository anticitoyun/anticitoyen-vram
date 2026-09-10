# Le noyau d'attention au décodage : où nous en sommes

État au 9 septembre 2026, fin de soirée. **Écrit pour être repris sans le
contexte de la session**, et surtout pour que rien de faux ne survive.

## Ce qui est mesuré et tient

    lanceur CPU        0,1 % du temps      evenements CUDA, montage independant
    graphes CUDA       x2,17 sur un MoE    temps GPU identique avec et sans
    pente de contexte  -12,2 %             350 -> 3000 mots, dispersion 0,2 %
    llama.cpp          x1,96 au decodage   meme source, ABBA, protocole ecrit

## Ce qui a été retiré, et pourquoi

    46,5 us de cout fixe dans le noyau     RETIRE
    49,5 us par appel                      RETIRE
    94 % de la duree independante du travail  RETIRE
    l attention a 28,7-36,5 % du pas       A REVERIFIER (meme origine)

Tous venaient d'un micro-banc qui enchaînait **300 appels sur les mêmes
tampons de sortie**. Le GPU les sérialise : chaque appel attend le précédent.
**Le chiffre était une latence de file, pas un coût de travail.**

Ce qui l'a démasqué : la bisection (`ACVRAM_PA_ETAPE`). Un noyau qui écrit
trois flottants et sort coûte **le même temps** que le noyau complet —
49,4 contre 49,5 µs. Aucune étape coupable, donc le temps n'était pas dans le
corps du noyau.

## Ce qui n'est PAS réfuté, contrairement à ce qui a été annoncé

    sous-parallelisation (0,38 bloc par SM)   NON TESTEE
    allocations par appel                     NON TESTEE
    lancement sous graphes                    NON TESTEE

Le balayage de grille qui les « réfutait » rend les mêmes chiffres au dixième
près que la bisection à l'étape 0 : **même montage, même latence**. Une grille
multipliée par huit ne pouvait pas faire bouger un chiffre qui ne mesurait pas
le travail.

**Une réfutation fondée sur une absence de variation exige le contrôle qu'on
applique partout ailleurs : quelque chose doit changer quand on bouge le
paramètre.** Un temps invariant sur un facteur 64 de grille est aussi
compatible avec une grille *inerte* qu'avec un noyau insensible.

## L'outillage est prêt, c'est le protocole qui manque

* `ACVRAM_PA_ETAPE` (0-3) — sortie anticipée du noyau, sorties neutres écrites
  et dépendantes de `sq`/`sm` pour que rien ne soit supprimé ;
* `ACVRAM_PA_CHUNK` — taille de tranche à l'exécution, donc grille variable
  sans recompiler ;
* le contrôle d'empreinte — le binaire doit porter le sha du source, sinon
  refus : `ccache` rendait des objets périmés sans que rien ne le dise.

**Ce qu'il faut écrire : un protocole qui n'enchaîne pas des appels partageant
leurs tampons.** Tampons distincts par appel, ou appels réellement
indépendants — sinon on remesure la file.

## Un candidat à vérifier en premier

**Un parcours de taille fixe** : un noyau qui balaierait toute la table de
pages, ou une boucle bornée par une capacité plutôt que par l'occupation
réelle. Cela expliquerait à la fois un temps indépendant du travail et une
insensibilité à la grille, chaque bloc refaisant le même balayage. **La
bisection le désignera immédiatement, une fois le protocole réparé.**

## Le protocole de reprise (écrit le 10/09, avant toute mesure)

`outils/attn-isole.py`. **Un seul appel entre deux événements CUDA, puis
synchronisation.**

Pourquoi cela corrige le défaut : les événements mesurent le temps GPU **entre
les marqueurs**, donc la synchronisation qui suit n'entre pas dans
l'intervalle — contrairement à un chronomètre mur, où elle ajoutait ~9 µs par
appel et nous avait déjà coûté une mesure. Et comme un seul appel est en vol,
**aucune file ne se forme** : plus de sérialisation artificielle.

**Ce qui ne suffisait pas** : passer `ACVRAM_PAGED_ALLOC=1` pour retrouver des
`torch::empty` par appel ne change rien — l'allocateur CUDA de PyTorch rend le
**même bloc** à chaque fois, donc les appels écriraient encore au même endroit.
Ce n'est pas l'allocation qu'il fallait séparer, c'est la file.

### Le témoin est la bisection elle-même

    ACVRAM_PA_ETAPE=0   indices + trois flottants ecrits, retour
    ACVRAM_PA_ETAPE=3   noyau complet

**L'étape 0 doit coûter beaucoup moins que l'étape 3.** Si les deux rendent le
même temps, le montage est encore faux et **rien de ce qu'il mesure ne vaut** —
c'est exactement ce qui a disqualifié le précédent. Le montage précédent était
réfuté par la bisection ; celui-ci est validé par elle, avant toute lecture.

### Ordre imposé

    1. empreinte : le .so porte-t-il le sha du .cu ? (ccache ment par la date)
    2. temoin du montage : etape 0 contre etape 3
    3. bisection complete, deux contextes
    4. grille (ACVRAM_PA_CHUNK), sur un montage enfin valide

Cet ordre est **exécuté**, pas rappelé : `outils/protocole-49us.sh`. Le témoin
y est une garde bloquante — verdict `PLAT`, sortie 2, aucune ligne suivante
n'est lue. Chaque mesure prévient à sa fin (pas seulement à la fin du lot),
refuse de démarrer si la carte n'est pas prenable (`outils/carte-libre.sh`,
partagé avec tout le circuit), et passe par `timeout` : aucune
mesure ne peut retenir la machine.

**La grille vient en dernier et pas avant** : c'est elle qui a produit la
fausse réfutation de la sous-parallélisation, et elle ne voudra dire quelque
chose que sur un montage dont le témoin a parlé.

## Le plancher du harnais (10/09) — et une cause à ne pas retenir

**La disqualification est plus large qu'annoncé** (poste1) : le montage rendait
~49,5 µs **quel que soit le noyau**, donc les « 46,5 µs de coût fixe, 94 % de
la durée » ne sont pas une propriété du noyau — **un plancher est par
construction indépendant du travail, et c'est exactement ce que la
décomposition a mesuré.** La pente `28,0 + 0,0234·n` retourne en « non
mesuré ». Ce qui survit : les 49,5 % du temps GPU, pris au profileur sur
exécution réelle — autre instrument, autre population.

### La cause proposée ne tient pas, et il faut le savoir avant de « corriger »

Le diagnostic était « sans doute une synchronisation par appel », avec pour
correctif : événements autour du lot, une seule synchronisation, division par
le nombre d'appels. **Vérification faite dans le code, les deux montages
fautifs le faisaient déjà** :

    e0.record(); for _ in range(300): appel(); e1.record()
    torch.cuda.synchronize(); us = e0.elapsed_time(e1)*1000/300

**Le correctif était donc déjà en place, et l'appliquer n'aurait rien changé —
on aurait remesuré le même plancher en croyant l'avoir supprimé.** La cause
reste inconnue.

### Ce qui rend cela sans importance : on chiffre le plancher au lieu de le déduire

Le plancher est chiffré **par l'étape 0 elle-même**, rejouée aux bornes du
balayage réel : un noyau qui écrit trois flottants et sort ne mesure rien
d'autre que le harnais.

**Le balayage `K` de `banc_fma` a été retiré** (`dbac2d1`) : il chiffrait le
même plancher avec une **autre** configuration de lancement, donc un plancher
qui n'était pas celui du noyau mesuré. Le noyau `banc_fma_kernel` reste dans le
`.cu`, inutilisé et sans coût ; seul le script serait à réécrire.

**Il redevient nécessaire dans un seul cas, et ce cas est écrit d'avance :**

    etape 0 << etape 3   -> le montage a rendu DEUX valeurs differentes : il a
                            prouve DE LUI-MEME qu'il peut rendre autre chose.
                            La bisection EST le controle positif. K inutile.

    etape 0 ~= etape 3   -> montage encore faux, arret sans rien lire. C'est LA
                            que K devient necessaire : il faut alors une echelle
                            de travail CONNUE pour savoir si l'instrument est
                            aveugle ou si le noyau est vide.

`banc_fma` est donc l'outil du **cas d'échec**, pas du cas nominal. Le seuil du
témoin est inscrit dans `outils/protocole-49us.sh` avant toute mesure : séparé
si la médiane de l'étape 0 est sous 70 % de celle de l'étape 3 **et** que
l'écart dépasse les dispersions cumulées. Deux conditions, parce que la seconde
seule laisserait passer une séparation minuscule mais régulière, et la première
seule un écart franc noyé dans le bruit.

### Le critère de participation ne se lit pas dans un temps

Un temps plat est compatible avec « noyau insensible au parallélisme » **et**
avec « grille inerte ». On observe donc qui a tourné : à l'étape 0, chaque bloc
écrit un marqueur dans sa case de `part_m`, et le compte des cases marquées se
relit côté hôte.

    participants == grille, temps varie   -> mecanisme vivant
    participants == grille, temps plat    -> refute pour de bon
    participants <  grille                -> GRILLE INERTE : le lancement est
                                             en cause, rien n'a ete teste
    participants <  grille, temps varie   -> contradiction, instrument en cause

### Règle qui vaut désormais pour tout micro-banc

**Aucun temps par noyau ne se publie sans le plancher de son harnais, mesuré
par le cas vide, à côté de lui.** Un temps sous ~5 fois ce plancher se publie
comme « sous le plancher de l'instrument », jamais comme une valeur.

## Ce que l'égalité voudra dire — écrit AVANT la mesure

Le plancher (étape 0) est rejoué **à chaque valeur de grille**. Deux issues, et
**les deux sont des renseignements** :

    plancher qui VARIE avec la grille
      -> il est dans le lancement des blocs ; la grille agit sur quelque chose,
         et la sous-parallelisation redevient testable par difference

    plancher IDENTIQUE a toutes les grilles
      -> il n'est PAS dans le lancement des blocs. Il est dans le chemin commun
         a tous les appels : Python -> C++ -> pilote. C'est le PREMIER
         renseignement sur la cause inconnue, et il vaut plus que le choix d'un
         harnais.

**Cette phrase est écrite d'avance exprès.** Sans elle, une égalité se lirait
« le test n'a rien donné » — la faute commise quatre fois le 9/09, où une
absence de variation a été prise tantôt pour une réfutation, tantôt pour un
échec, jamais pour ce qu'elle disait.

### Règle de départage, pré-inscrite

Si deux protocoles rendent des planchers **non distinguables** — écart inférieur
à leur dispersion combinée —, **on ne choisit pas par la mesure**. On garde
l'**appel isolé**, pour une raison posée d'avance et indépendante des chiffres :
c'est le seul des deux qui **ne peut pas** masquer un effet de file, et la file
est le mode de défaillance effectivement observé. Un départage doit être un
principe pré-inscrit, jamais une préférence formée après avoir vu les nombres.

### Ce qui est déjà satisfait dans `outils/attn-isole.py`

* dispersion publiée avec chaque valeur : médiane de 51, p10 et p90 ;
* plancher mesuré **aux bornes du balayage réel**, puisque l'étape 0 est
  rejouée pour chaque valeur de `ACVRAM_PA_CHUNK` ;
* participation **observée** par atomique, jamais reconstruite.

## Le résultat du 10/09 : le noyau est sérialisé sur la longueur de tranche

Protocole `outils/protocole-49us.sh`, 12 mesures, ordre imposé respecté.

### Le témoin a séparé — le montage est valide

    etape 0 (indices)   10,08 us [9,73 - 10,59]
    etape 3 (complet)   55,46 us [54,82 - 56,54]     -> SEPARE

Facteur 5,4, là où l'ancien montage rendait 49,5 des deux côtés. **Le plancher
du harnais vaut ~10 µs, pas 46,5.** Le balayage `K` de `banc_fma` reste inutile :
la bisection a prouvé d'elle-même qu'elle pouvait rendre deux valeurs.

### La bisection : tout le coût est dans la boucle principale

    etape          ctx 357    ctx 3007
    0 indices       10,05       10,88
    1 + q charge    10,08       10,98
    2 + boucle      55,20       78,46
    3 + reduction   55,39       78,72

Le saut est entre 1 et 2, et il **dépend du travail** (+23 µs de 357 à 3007).
La réduction finale coûte 0,2 µs : rien. Les « 94 % indépendants du travail »
sont morts pour de bon.

### La grille : le temps double quand la tranche double

Contexte 3007, étape 0 rejouée à chaque valeur.

    chunk      C   e0(us)   e3(us)  travail   participants
       64     64    15,36    25,47    10,11   1504/1504 = utiles
      128     32    11,52    31,58    20,06    768/ 768 = utiles
      256     16    11,55    45,92    34,37    384/ 384 = utiles
      512      8    11,81    78,72    66,91    192/ 192 = utiles
     1024      4    11,62   144,45   132,83     96/  96 = utiles
     2048      2    11,36   279,36   268,00     64/  64 = utiles

**Le total vaut le temps d'UNE tranche.** Découper davantage le réduit d'autant :
`chunk=64` coûte 25,47 µs contre 78,72 en production, soit **×3,1 sur un noyau
qui pèse 49,5 % du temps GPU**.

### Ce que l'égalité voulait dire : ni l'une ni l'autre des deux issues

Le plancher est **plat à ~11,5 µs sur un facteur 12 de grille** (64 → 768
blocs), puis monte à 15,36 µs à 1504 blocs. Les deux issues écrites d'avance
étaient trop tranchées : le plancher est dans le chemin commun Python → C++ →
pilote, **plus** une composante d'occupation qui n'apparaît qu'au-delà de
~768 blocs. Seule une mesure à plusieurs grilles pouvait le dire — une seule
valeur aurait donné l'une des deux réponses fausses.

### Le contrôle qui se trompait dans le sens de la prudence

Le compteur de participation comparait aux tranches **demandées** par la
capacité de la table (`32·C`), pas aux tranches **utiles** du contexte réel.
Il aurait affiché « GRILLE INERTE » à cinq valeurs de `chunk` sur six et fait
jeter tout le balayage. Recalculé sur `ceil(seq_len/chunk)`, il est exact
partout. **Un contrôle peut se tromper dans le sens de la prudence, et c'est le
sens dans lequel personne ne va vérifier.**

## Ce qui est attendu au moteur — ÉCRIT AVANT LA MESURE

La tranche adaptative prend la plus petite tranche qui garde `C ≤ 256`, la
borne étant ensuite vérifiée et non supposée. 64 est le plancher parce que
c'est la plus petite valeur **mesurée** : 32 et 16 n'ont pas été essayés.

    N*16 jetons    chunk choisi   C
        512            64          8
       8192            64        128
      16384            64        256
      32768           128        256
      65536           256        256

**Prédiction** : noyau ×3,1 sur un poste à 49,5 % du temps GPU donne
`1/(0,505 + 0,495/3,1) ≈ 1,50`, soit **+50 % de débit de décodage**.

Ce que chaque issue voudra dire, posé d'avance :

    ~ +50 %          le gain se transporte, le modele d Amdahl tient
    nettement moins  le temps gagne est REPRIS AILLEURS, et il faudra dire ou
                     avant d annoncer quoi que ce soit
    ~ 0 %            DEUX causes possibles, a departager et non a supposer :
                     (a) le temps gagne est REPRIS par un poste que le noyau
                         isole ne voyait pas — c est l hypothese a tester EN
                         PREMIER, car un banc de bout en bout paie ce que
                         l appel isole ne paie pas ;
                     (b) seulement si (a) est ecarte : les 49,5 % du profileur
                         mesurent autre chose que ce que l on croit.

### La borne mémoire dit que le gain n'est pas repris par l'attention elle-même

Qwen3-Coder-30B-A3B, cache int8 : 4 têtes KV × 128 = 512 octets pour K, autant
pour V, soit ~1032 octets par jeton et par couche (échelles comprises). À 3007
jetons, une couche relit **3,10 Mo** par pas ; à la borne mesurée de la 5090
(~1050 Go/s) cela vaut **2,96 µs**.

    noyau a chunk 512    78,72 us    26,6 x la borne memoire
    noyau a chunk  64    25,47 us     8,6 x la borne memoire

**Même après le gain, le noyau reste à 8,6 fois sa propre borne mémoire.** Le
temps gagné ne peut donc pas être repris par la bande passante de l'attention :
si le gain moteur est faible, la cause est ailleurs dans le pas, et il reste de
la marge sur ce noyau au-delà de ×3,1.

**Deux réserves écrites d'avance, elles aussi.** La grille n'a été balayée qu'à
**un seul contexte** (3007) : le gain à contexte court est attendu mais non
mesuré, et le banc doit couvrir les deux. Et un découpage qui change **change
l'ordre des sommes** : la barrière de qualité est obligatoire avant toute
annonce, témoin négatif compris.

## La validation moteur : le gain se transporte, et au-delà

ABBA au niveau du **processus** (voir plus bas pourquoi), quatre bras par
contexte, `b0c1c0b`, binaire `46b9760dd521cc5b`.

    350 mots    A 512   120,65 / 120,86 j/s    noyau 71,52 / 71,62 us
                B auto  227,45 / 226,97 j/s    noyau 19,14 / 18,98 us   +88,0 %
    3000 mots   A 512   105,69 / 105,72 j/s    noyau 78,72 / 78,85 us
                B auto  216,10 / 216,06 j/s    noyau 25,86 / 25,54 us  +104,4 %

Les deux passages de chaque bras se reproduisent à 0,2 %, l'ordre de passage ne
déplace rien, et le témoin noyau retrouve **exactement** les valeurs du
balayage isolé (25,47 et 78,72 µs).

### Le banc qui rendait +0,1 % portait un gain de +88 %

Première version : les deux bras alternaient **dans le même processus**, pour
qu'aucune différence de chargement ne s'y glisse. Or le décodage passe par un
**graphe CUDA capturé une fois**, et un graphe enregistre la grille
`dim3 g1(BQ, HQ, C)` au moment de la capture. Changer `ACVRAM_PA_CHUNK` ensuite
ne rejoue pas un nouveau lancement : il rejoue celui qui a été capturé.

**Le montage choisi pour éliminer une différence parasite avait éliminé la
différence étudiée.** Résultat : +0,1 % aux deux contextes, avec une étendue de
0,2 % sur le bras inerte contre 7,1 % sur l'autre — la signature d'un bras qui
n'a jamais tourné, lisible seulement si on la cherche.

Sans le témoin noyau ajouté ensuite, la seule lecture possible était « le gain
ne se transporte pas » : un résultat nul crédible, publié de bonne foi, sur un
poste qui **double** le débit. **Dix lignes de témoin contre un chantier
abandonné à tort.**

### Ce que le gain implique — et ce qu'il ne permet PAS de conclure

Le gain dépasse la prédiction de +50 %. **Un gain au-dessus demande la même
explication qu'un gain en dessous**, et les 49,5 % du profileur deviennent le
suspect.

**La conversion du gain en part d'attention est retirée.** L'équation d'Amdahl
attribuerait tout le gain à l'accélération du noyau, alors que passer de 512 à
64 multiplie aussi `C` par huit — donc le nombre de lancements et le
parallélisme. Et le facteur employé dépend d'un choix de dénominateur non
discuté : rapport des temps mesurés (3,04) ou rapport des **travaux**, plancher
déduit (66,91 / 10,11 = 6,6), qui donne 57 % au lieu de 76 %. Trois grandeurs
changent, l'équation en résout une. **Le chiffre viendra d'une mesure directe,
pas d'une inversion de formule.**

## La barrière de qualité : trois instruments avant d'en trouver un valide

### 1. `acvram eval` — aveugle à ce qu'il devait tester

Quatre passes, quatre fois **8,825** au millième. Ce n'était pas la neutralité :
`evaluate.py:189` appelle `model(batch, ...)`, le **forward dense**, qui
n'appelle jamais `paged_attention`. Une barrière aveugle à ce qu'elle teste rend
toujours « conforme ».

### 2. La comparaison de trajectoires — ne peut pas trancher

À température 0, les 128 jetons sont identiques à 350 mots, et divergent **au
jeton 24** à 3000. Deux implémentations également correctes divergent en
génération gloutonne : l'arrondi s'amplifie de façon chaotique. Ni preuve de
neutralité, ni preuve de dégradation.

*(Premier essai jeté : l'invite était faite d'identifiants `crc32`. Acceptable
pour un débit — le coût d'un pas ne dépend pas du sens — mais sur du charabia le
modèle part en répétition, et deux boucles dégénérées divergent entre candidats
quasi équiprobables sans rien dire du noyau.)*

### 3. La distance à l'attention dense — celle qui tranche

Comparer les deux découpages **entre eux** ne dit pas lequel est juste. Seul un
tiers qui ne partage pas leur défaut peut arbitrer : `decode_attention_fixed`,
qui ne découpe rien.

    350 mots    dense <-> 512  5,025485e-03    dense <-> adaptatif  5,025486e-03
                ecart entre les deux decoupages  4,54e-05  =  0,90 % de l'ecart
                deja accepte par la quantification du cache
    3000 mots   dense <-> 512  7,771520e-03    dense <-> adaptatif  7,771520e-03
                ecart entre les deux decoupages  3,60e-05  =  0,46 %

**Les deux découpages sont à la même distance de la référence** (sept chiffres
identiques à 3000). L'écart entre eux vaut moins de 1 % de l'erreur que la
quantification int8 du cache fait déjà subir. **Classe B, sans dégradation
mesurable** : le noyau est par ailleurs déterministe — deux appels identiques
rendent le bit exact.

## Ce qui reste à mesurer, avec sa prédiction écrite d'avance

**64 est un plancher d'ignorance, pas un optimum** : c'est la plus petite
valeur essayée. 32 et 16 n'ont jamais tourné, et on ne règle pas un défaut par
extrapolation.

Le modèle que les six points existants suggèrent — à vérifier, pas à croire :

    total(chunk) = plancher(nombre de blocs) + travail(chunk)

Le travail est proportionnel à la tranche (mesuré : il double quand elle
double). Le plancher croît avec le nombre de blocs, mais **seulement au-delà de
~768** : plat à 11,5 µs de 64 à 768 blocs, puis 15,36 µs à 1504.

    chunk 64    1504 blocs   plancher 15,36   travail 10,11   total 25,47 (mesure)
    chunk 32    3008 blocs   plancher   ?     travail  ~5     total  ?
    chunk 16    6016 blocs   plancher   ?     travail ~2,5    total  ?

**Prédiction** : le travail économisé (~5 µs de 64 à 32) est du même ordre que
la croissance du plancher observée en doublant les blocs (~3,8 µs de 768 à
1504). **On attend donc un gain nul ou faible à 32, et une perte à 16.** Si 32
gagne nettement plus de 5 µs, le modèle additif est faux et c'est lui qu'il
faudra reprendre — pas le réglage.

**Contrainte à ne pas oublier** : `C ≤ 256` limite `chunk=32` aux contextes
sous 8 k jetons et `chunk=16` sous 4 k. Un réglage qui ne vaut que pour les
contextes courts n'a d'intérêt que si le gain y est net.

**Et la marge est réelle** : à 3007 jetons le noyau reste à **8,6 fois sa borne
mémoire** après le gain (25,47 µs contre 2,96 µs pour relire 3,10 Mo à
1050 Go/s). Ce qui limite n'est donc toujours pas la bande passante.

## Le balayage 32/16 : 64 n'est plus un plancher d'ignorance

Contexte 3000, étape 0 rejouée à chaque valeur, `64` **encadrant** le balayage.

    chunk         plancher   total    travail    contre 64
    64  (debut)     15,36     25,66    10,30        —
    32              21,47     31,78    10,31     +23,9 %
    16              33,73     41,86     8,13     +63,1 %
    64  (fin)       15,36     25,57    10,21      -0,35 %

**Le `64` de fermeture rend 15,36 µs comme celui d'ouverture** — dérive nulle
sur le plancher, 0,35 % sur le total. Les trois valeurs se comparent.

**Prédiction vérifiée sur l'issue, réfutée sur le mécanisme.** J'attendais une
perte, et il y a bien perte aux deux valeurs. Mais je l'expliquais par un
plancher qui rattrape un travail décroissant ; or **le travail ne décroît
plus** : 10,30 µs à 64, 10,31 à 32 — identique au centième.

**La proportionnalité cesse en dessous de 64.** De 128 à 2048 le temps doublait
avec la tranche ; à 32 une tranche ne fournit plus assez de travail pour
occuper un bloc, et seul le plancher continue de croître — il double de 64 à 16
(15,36 → 33,73) pendant que le travail stagne.

Le modèle additif `plancher(blocs) + travail(chunk)` tient donc ; c'est son
second terme qui **sature**. Un modèle juste sur la forme et faux sur le
domaine de validité rend la bonne réponse pour la mauvaise raison — ce qui ne
se voit que si l'on mesure les deux termes séparément, et c'est ce que l'étape 0
rejouée à chaque valeur permet.

**64 est l'optimum des sept valeurs essayées**, et ce n'est plus un plancher
d'ignorance : les deux valeurs en dessous ont été mesurées et perdent.

## Le « plancher » n'est pas un plancher : il est expliqué, et c'est du travail

Trois mesures, chacune écartant une cause, dans cet ordre.

### 1. Ce n'est pas le lancement — `banc_fma`, à travail par bloc constant

    170 -> 4080 blocs     7,1 us        PLAT sur un facteur 24
    5440 blocs            8,74 us
    8160 blocs           10,75 us
    multiples de SM (1x, 2x, 4x), +/-2 blocs : aucune marche

**Ni vagues ni débit de distribution.** L'hypothèse des vagues prédisait des
marches au franchissement de la capacité résidente : il n'y en a aucune. Le
lancement de 1504 blocs coûte 7,1 µs, là où l'étape 0 en coûte 15,3 **au même
nombre de blocs**. La différence n'est donc pas dans le lancement.

*(C'est `banc_fma` qui a tranché — le noyau gardé « pour le cas d'échec » et
retiré du protocole. Il a servi à un cas nominal que personne n'avait prévu.)*

### 2. Ce n'est pas mon compteur — hypothèse à moi, réfutée par moi

L'étape 0 fait un `atomicAdd` par bloc **sur une adresse unique** : tous les
blocs frappent la même case, le L2 les sérialise, et le coût croît avec leur
nombre. La forme était la bonne, l'amplitude non :

    chunk 64  (1504 blocs)   15,23 allume  /  15,30 eteint   0, dans le bruit
    chunk 16  (6016 blocs)   34,30 allume  /  32,26 eteint   2,04 us, soit 6 %

**0,34 ns par bloc.** Il fallait en expliquer 18 µs. Le compteur est disculpé —
et il ne pouvait l'être que parce qu'on peut désormais l'éteindre
(`ACVRAM_PA_SANS_COMPTEUR=1`). **Un instrument qui ne peut pas être éteint ne
peut pas être disculpé.**

### 3. Ce que c'est : l'écriture des tampons partiels

    plancher = 9,65 us + 3,76 ns par bloc

    1504 blocs   mesure 15,30   modele 15,30   ecart 0,0 %
    3008 blocs   mesure 21,47   modele 20,95   ecart 2,4 %
    6016 blocs   mesure 32,26   modele 32,26   ecart 0,0 %

Le terme fixe (9,65 µs) est du même ordre que le lancement mesuré par
`banc_fma` (7,1 µs). Le terme par bloc correspond aux **512 octets** que chaque
bloc écrit dans `part[]` (128 flottants) : 3,76 ns pour 512 octets font
**136 Go/s effectifs**, un débit d'écriture dispersée plausible.

**Il n'y a donc plus de mystère matériel.** Ce que nous appelions « plancher »
depuis hier est, pour les deux tiers, **du travail réel** : écrire les
résultats partiels. Et cela a une conséquence directe sur le réglage :

> **Diviser la tranche par deux double le nombre de blocs, donc double les
> octets de partiels écrits.** C'est ce terme qui remonte quand `chunk`
> descend, et c'est lui — pas une capacité, pas un ordonnanceur — qui fait
> perdre 32 et 16.

Le modèle additif posé plus haut se referme : `travail(chunk)` sature en
dessous de 64, tandis que le second terme, mal nommé « plancher », **croît
avec le nombre de blocs parce qu'il écrit un partiel par bloc**.
