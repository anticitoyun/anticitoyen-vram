# Contrôle positif de la grille, et critère écrit d'avance

poste1, 10/09/2026, **avant toute mesure**. Objet : la sous-parallélisation de
`paged_attn_partial` repasse en « ouvert » après disqualification du micro-banc.

## 0. La disqualification est plus large qu'annoncé

Le montage rendait ~49,5 µs par appel **quel que soit le noyau** — y compris un
noyau qui écrit trois flottants et sort. Ce n'est pas un temps de noyau, c'est
le **plancher de l'instrument** : un coût par appel payé côté hôte, sans doute
une synchronisation par appel.

Or ce même 49,5 apparaît dans **deux** conclusions :

    balayage de grille   49,6 · 49,5 · 71,9 · 71,7   -> declare « insensible »
    decomposition        46,5 de fixe sur 49,5 total -> declare « 94 % fixe »

**Le « 94 % du temps indépendant du travail » est donc le plancher de
l'instrument, pas une propriété du noyau.** Un plancher est par construction
indépendant du travail : c'est exactement ce que la décomposition a mesuré.

Ce qui survit, et seulement cela : **la part de 49,5 % du temps GPU**, obtenue
au profileur sur une exécution réelle — un autre instrument, une autre
population. La décomposition en coût fixe et la pente en contexte retournent à
« non mesuré », au même titre que la sous-parallélisation.

## 1. Le contrôle positif : une échelle de travail

Un instrument ne rend un résultat que s'il pouvait en rendre un autre. On lui
donne donc une différence **connue d'avance et grande**, et il n'est déclaré
valide que s'il la retrouve.

Même harnais, même grille, même noyau ; un seul paramètre : `K` itérations
d'une boucle de FMA **dépendantes** (chaînées, pour qu'aucune ne soit éliminée
par le compilateur ni recouverte).

    K = 1, 10, 100, 1000, 10000

* le temps doit devenir **linéaire en K** dès que le travail dépasse le
  plancher ;
* **le coude désigne le plancher** : la valeur de K à partir de laquelle le
  temps bouge donne le plancher en microsecondes, sans hypothèse.

Ce balayage fait donc les deux choses à la fois : il prouve que l'instrument
peut bouger, et il **chiffre** ce qu'il ne peut pas voir. Aucun autre contrôle
n'est nécessaire avant lui.

**Correctif de harnais à appliquer en même temps** : une seule synchronisation
pour la série entière, événements CUDA autour du lot et non de chaque appel,
division par le nombre d'appels — et, si possible, la série capturée sous
graphe. Un plancher de 49,5 µs par appel est le signe d'une synchronisation par
appel ; sans ce correctif, l'échelle de travail mesurera le plancher jusqu'à
K = 10000.

## 2. Le critère qui distingue « grille inerte » de « noyau insensible »

**Il ne se lit pas dans un temps.** Un temps plat est compatible avec les deux,
et c'est précisément le piège où le montage est tombé. On observe donc
directement **qui a participé**.

Un compteur de participation : chaque bloc écrit son `blockIdx` dans un
emplacement, ou incrémente atomiquement un compteur. On relit après
l'exécution. Coût : un tampon, aucune mesure de temps.

Table de décision, posée maintenant :

| participants relus | le temps varie avec la grille | conclusion |
|---|---|---|
| = grille demandée | oui | grille active, noyau sensible → **la sous-parallélisation est vivante et mesurable** |
| = grille demandée | non | grille active, **noyau réellement insensible** → mécanisme réfuté, cette fois pour de bon |
| < grille demandée, constant | non | **grille inerte** → non testé ; réparer le lancement avant toute conclusion |
| < grille demandée | oui | contradiction → l'instrument est en cause, on ne conclut rien |

La troisième ligne est l'hypothèse qui rendrait le résultat d'hier nul : les
partitions recalculées en interne depuis la longueur de contexte, les blocs
surnuméraires sortant immédiatement.

## 3. Règle de publication pour tout micro-banc

> Aucun temps par noyau ne se publie sans le **plancher de son harnais**, mesuré
> par le cas vide, à côté de lui. Et un temps inférieur à ~5 fois ce plancher se
> publie comme **« sous le plancher de l'instrument »**, jamais comme une valeur.

C'est la garde qui aurait arrêté les trois conclusions d'hier en une ligne : le
noyau vide et le noyau complet rendaient le même chiffre, et personne n'avait
mesuré le noyau vide.
