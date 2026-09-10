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
