# Decodage MLA par lot : ce qu'il faudrait, et ce que ca vaut

## EN TETE, pour qu'on ne parte pas sur MLA en croyant tenir le goulot

    lancements par pas, GLM-4.7, b = 12, 66 couches
      total mesure par nsys        36 486
      dont mla_decode (ce compte)   1 584   =  4,3 %
      reste                        34 902   = 95,7 %

**Le decodage MLA par lot ne toucherait que 4,3 % des lancements**, et mon
estimation de ~6 % du temps sous graphes est coherente avec ce compte. **Le
gisement n'est pas la.** Par couche : 553 lancements, dont 24 pour MLA — il en
reste **529 par couche a expliquer**, et le facteur 3,8 cherche dans MLA n'y est
pas.

Ce qui justifie quand meme ce chantier n'est pas un pourcentage de lancements,
c'est la levee de la limite de creneaux et de la reservation d'un `max_len`
entier par creneau — plus, si le chemin par creneaux calcule faux, le fait que
la pagination cesse d'etre une optimisation pour devenir la CORRECTION.


poste4, 10/09/2026. **Lecture seule** de `acvram/engine/mla.py` et de
`acvram/kernels/acvram_kernels.cu:2341-2371`. Aucune mesure, aucun code.

## 1. Ce qui est par sequence, ce qui est commun

`mla_decode(q_eff, cache, len, scores, L, rank, scale)` lance **deux** noyaux :

    mla_scores_kernel<<< dim3(H, ceil(L/128)), 128 >>>
    mla_reduce_kernel<<< H, 256, L*sizeof(float) >>>

    PAR SEQUENCE   q_eff [H, W]   cache [L, W]   len (scalaire)
                   scores [H, L]  (tampon de travail)   sortie o [H, rank]
    COMMUN         L (godet), W, rank, scale
    DEJA PAR LOT   k_b et v_b : appliques en Python par `einsum` hors du
                   noyau (mla.py:281 et 291). Ils ne sont pas dans l'obstacle.

**Il ne reste donc que quatre tenseurs a doter d'un axe de lot**, et trois
d'entre eux sont deja des tampons persistants par creneau. Le seul argument
reellement commun qui pourrait genner est `len` : il devient `[b]` au lieu d'un
scalaire, ce que `mla_scores_kernel` lit deja par pointeur (`len.data_ptr<long>()`).

## 2. La forme de grille suffit-elle ?

**Pour le calcul, oui** : un axe `z` de grille, `blockIdx.z` = la sequence, et
les deux noyaux deviennent `dim3(H, ceil(L/128), b)` et `dim3(H, 1, b)`.

**L'obstacle n'est pas la grille, c'est l'ADRESSAGE du cache.**
`new_static` (mla.py:245) alloue **un tenseur contigu par creneau** :
`cache [max_len, rank+rope]`, `scores [nh, max_len]`. A `b` creneaux, ce sont
`b` bases distinctes — un noyau ne peut pas les atteindre par un simple
`blockIdx.z` sans qu'on lui passe un tableau de pointeurs.

Deux sorties, et la premiere est nettement meilleure :

- **une arene unique** `[slots, max_len, W]` (et `[slots, nh, max_len]` pour
  les scores), le creneau devenant la premiere dimension. L'adresse redevient
  `base + z * max_len * W`, aucun tableau de pointeurs, aucun acces indirect.
  C'est un changement d'ALLOCATION, pas de noyau ;
- un tableau de pointeurs `const bf16* const*` : marche, mais ajoute une
  indirection par bloc et un transfert hote->carte par pas.

Contrainte a ne pas perdre de vue : `mla_reduce_kernel` demande **`L` flottants
de memoire partagee** par bloc, et le lanceur releve la limite au-dela de
48 Kio. Le passage au lot **ne l'aggrave pas** — un bloc reste
(une tete, une sequence) — mais il ne la resout pas non plus : le godet reste
borne par la memoire partagee, ~24 Ki positions au mieux sur sm_120.

## 3. Le rapprochement avec l'attention paginee

**Le cache latent a deja la bonne forme.** Il est `[L, rank+rope]`, soit un
vecteur par jeton — exactement ce qu'une table de blocs adresse
(`[num_blocks, block_size, W]`). Et l'ecriture est deja ponctuelle :
`cache.index_copy_(0, st["len"], k_new)` ecrit **une ligne a la position
`len`** ; pagine, elle ecrirait a la position donnee par `slot_mapping`,
exactement ce que fait deja `kv_write_int8_kernel`.

**Ce qui s'y oppose n'est donc pas la geometrie, c'est la plomberie de l'etat.**
`mla.py` tient son propre magasin (`st`, creneaux `_STATIC`), separe de
`m.caches` et de l'allocateur de blocs. Et le chargeur ne pagine pas les
couches MLA : leur branche fait `continue` avant `a_allouer.append` — releve
le 10/09 en corrigeant le budget KV, et teste dans
`tests/test_budget_kv_hybride.py::test_mla_stocke_sans_paginer`. **Une couche
MLA stocke sans paginer, par construction.**

**Mon avis, appuye sur ce qui precede : remplacer, pas reparer.** Pagine, le
cache latent perd d'un coup (a) la boucle par sequence, (b) la limite de
creneaux — donc le defaut de corruption au-dela de quatre devient sans objet
plutot que d'etre corrige, (c) la reservation d'un `max_len` entier par
creneau, qui est ce qui rend les creneaux couteux et donc rares. Le travail est
dans le chargeur et dans `mla.py`, pas dans le `.cu` : le noyau a besoin d'un
axe de grille et d'une base indexee, ce qui est le plus petit des trois.

## 4. Ce que ca vaudrait — ordre de grandeur, avec les chiffres qu'on a

A `b = 12` et 66 couches, et **deux lancements par appel** :

    par sequence   66 x 12 x 2 = 1 584 lancements par pas
    par lot        66 x 2      =   132
    economie                     1 452 lancements par pas

Au cout de noeud mesure par poste2 :

    sous graphes   1 452 x 0,61 us =  0,89 ms par pas
    en eager       1 452 x 2,40 us =  3,49 ms par pas

Rapporte au pas mesure a douze sequences (**14,55 ms**, mon tableau du 10/09) :

    sous graphes   ~6 %
    en eager       ~24 %

**Je ne promets donc pas un facteur deux, et il faut le dire avant que
quelqu'un l'espere.** Le compte de lancements borne le gain a ~6 % dans le
regime servi. Ce qui peut aller au-dela n'est pas borne par ce calcul :
occupation (chaque lancement actuel n'occupe que `H x ceil(L/128)` blocs,
soit une fraction des 170 SM), effets de queue, et surtout la **suppression de
la limite de creneaux**, dont le gain n'est pas un pourcentage mais une
concurrence servie plus grande.

**Ce qui est mesurable AVANT d'ecrire quoi que ce soit** : compter les noeuds
du graphe capture sur un MLA a b=1, 4, 12. Si le compte croit lineairement avec
b, la boucle est confirmee cote grave, et les 1 584 ci-dessus sont le bon
chiffre. Mon compteur de rejeux ne repond pas a cette question — un rejeu
execute tous les noeuds graves.

## Troisieme benefice, VERIFIE : la memoire partagee cesse de dependre de L

Je l'avais signale comme « a verifier » ; c'est verifie, dans notre propre
attention paginee (`acvram_kernels.cu:948-950`) :

    __shared__ float sq[D];
    __shared__ float sm[PA_WARPS], sl[PA_WARPS], scorr[PA_WARPS];
    __shared__ float sacc[PA_WARPS][D];

**Memoire partagee de taille FIXE**, fonction de `D` et de `PA_WARPS`, jamais de
la longueur de contexte — avec maximum et somme courants (`sm`, `sl`) et un
facteur de correction (`scorr`), c'est-a-dire un softmax EN LIGNE, bloc par
bloc. Le `L * sizeof(float)` de `mla_reduce_kernel` disparaitrait donc de
lui-meme en paginant : la reduction n'a plus besoin de voir tout le contexte a
la fois.

Le plafond de godet impose par la memoire partagee (~24 Ki positions sur
sm_120) tomberait avec lui. C'est un troisieme benefice que je n'avais pas
compte, et il ne coute rien de plus : le mecanisme existe deja dans le depot.
