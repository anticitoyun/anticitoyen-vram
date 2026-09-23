# Ce qui, dans le pas de decodage, n'a pas a y etre

Laurine, 10/09/2026. **Lecture du code, aucune mesure.** Chaque point dit ce
qui est etabli par le source et ce qui reste a verifier.

## Prealable : le profil a-t-il ete pris SANS graphes CUDA ?

**36 486 lancements par pas est un chiffre de mode EAGER.** Sous graphes, tout
le pas est rejoue en un lancement — c'est leur raison d'etre, et j'ai mesure le
10/09 que les couper coute +11,53 Gio de pic sur ce meme GLM-4.7. Avant de
depenser une refonte, **verifier que le profil portait graphes actifs** : si
`nsys` les a desactives, ou si une couche exilee les a supprimes, la plomberie
ci-dessous est deja amortie en production et le vrai sujet est ailleurs.

C'est la premiere chose a etablir, et elle se lit dans la sortie du profil :
`graphes actifs, N vivants` que je publie desormais dans mes bras.

## Le chemin reellement pris en decodage

`MoEBlock.forward` (model.py:860) : a `t <= _MOE_GROUPED_MAX` c'est
**`_forward_grouped`** (model.py:771). `_forward_prefill_grouped` et son
`_tuiles` ne servent qu'aux grands lots — donc **le `int(ntiles.sum())` de
model.py:705, qui est une synchronisation hote, n'est PAS dans le pas de
decodage.** Je le signale quand meme : c'est une synchronisation par couche et
par pas en prefill, et le prefill compte dans le TTFT.

## Ce qui peut sortir du pas, par ordre de facilite

**1. `tok = torch.arange(t).repeat_interleave(top_k)` — model.py:775.**
Ne depend que de `(t, top_k)`, tous deux CONSTANTS pour un lot de decodage
donne. Recalcule a chaque couche et a chaque pas. 66 couches x N pas de
`arange` + `repeat_interleave` identiques. **Precalculable une fois par valeur
de `t`**, dans un petit cache `{t: tenseur}`.

**2. `seq = torch.arange(eid.shape[0], dtype=int32)` — model.py:780 ET 798.**
Meme grandeur (`t * top_k`), **calcule DEUX FOIS par couche** dans la branche
avec porte. Et c'est la permutation IDENTITE : la passer a `_grouped` demande au
noyau d'indexer par `i -> i`. Deux gestes possibles, du moins au plus profond :
la precalculer comme (1), ou **accepter `None` cote noyau** pour dire « pas de
permutation » et supprimer l'indexation.

**3. `eid = topi.reshape(-1).to(torch.int32)` — model.py:774.**
Une conversion de type par couche et par pas. Le routage pourrait emettre
directement de l'int32 ; a verifier dans `_route`, je ne l'ai pas lu.

**4. `torch.cat(sorties)` — model.py:1010.**
Il est dans la **boucle par sequence** de la couche a attention lineaire
(`for i, ql in enumerate(batch.query_lens)`), c'est-a-dire le chemin GDN. Sur
un MLA pur comme GLM-4.7 cette boucle ne devrait pas s'executer — **a verifier
avant d'y toucher**, mais si elle s'execute, c'est elle qui fait a la fois le
`cat` et une part des lancements qui croissent avec la concurrence. Remede
nomme : ecrire chaque `y` dans une tranche d'un tampon prealloue au lieu de
concatener.

## Ce qui ne sort pas

- `xs = x[flat_t[ordre]]` et `d[inv]` (chemin prefill) : ce sont le rassemblement
  et la dispersion par expert. Ils dependent du routage, donc du contenu. Ils ne
  se hissent pas ; ils se **fusionnent** dans le noyau groupe, ce qui est un
  travail de `.cu`, pas de Python.
- `argsort`/`bincount` sur les indices d'experts : meme raison.

## Ce que je ne dis pas

Que ces quatre points expliquent les 85 % de trous. Un `arange` de douze
elements ne coute pas 10 us de GPU — il coute un **lancement**, et c'est le
lancement qui est en cause, pas le calcul. Le compte de lancements par pas
(36 486) est le chiffre a faire baisser ; le temps de ces noyaux est
negligeable. Confondre les deux ferait chercher des microsecondes la ou il faut
chercher des APPELS.
