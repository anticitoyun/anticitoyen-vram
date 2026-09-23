# La courbe du quota n'est pas plate — et son meilleur rendement est en haut

## Les quatre points

```
point                  bits/poids      PPL    vs etalon   PPL gagne/bit
plancher tout-nvfp4        4,7297   5,6102    +3,622 %          —
budget 4,50 Gio            5,7396   5,5643    +2,774 %       0,0455
budget 5,00 Gio            6,3737   5,5394    +2,314 %       0,0393
plafond tout-int8          8,3453   5,4144    +0,006 %       0,0634
```

Étalon extérieur `transformers`/GPTQ : **5,4141**, corpus `wiki-gptq.txt`
(sha `e52922746ad09bac`), 168 segments de 2048, vérifié identique au jeton.

## L'hypothèse de départ est réfutée

Le feu vert du matin disait : *« si elle est plate jusqu'à 5,5 bits, il y a des
octets à reprendre gratuitement »*. **Elle n'est plate nulle part**, et le coût
en octets d'un point de perplexité gagné ne croît pas — il **décroît** au
sommet :

```
4,730 -> 5,740 bits :  22,00 bits par point de PPL
5,740 -> 6,374 bits :  25,47
6,374 -> 8,345 bits :  15,77   <- le meilleur rendement
```

Il n'y a donc **pas d'octets gratuits à reprendre en bas de la courbe** : chaque
bit retiré coûte de la qualité, et le retirer là où on croyait la courbe plate
coûte *plus* cher par bit que de le retirer en haut.

**Réserve à porter avec ce résultat, et elle porte sur le dernier segment :** le
plafond n'est pas un point de budget, c'est un changement de **format** —
tout-int8, sans sac à dos. Son meilleur rendement peut donc venir du format et
non du quota. Les deux extrêmes doivent être **refaits par le protocole de
campagne** pour que la courbe soit homogène ; c'est ce que Laurine tient, et
c'est exactement pourquoi un témoin réutilisé n'est pas un témoin.

Seconde réserve : le plancher vient d'un relevé archivé, produit par un binaire
d'avant les changements du jour. L'écart de binaire mesuré sur le plafond vaut
0,0002 PPL (5,4142 archivé contre 5,4144 aujourd'hui), soit 0,4 % du plus petit
pas de la courbe — négligeable, mais dit plutôt que supposé.

## Chaque point est auditable depuis son propre dossier

```
                     plancher  plafond  depense  restant  promus
budget 4,50 Gio        3,7059   6,5464   4,4998  0,2 Mio  110/225
budget 5,00 Gio        3,7059   6,5464   4,9978  2,3 Mio  147/225
```

Le sac à dos dépense son budget **à 0,2 et 2,3 Mio près**. Sans les deux
silences nommés ce matin — budget sous le plancher, budget inemployé — un point
aurait pu être une conversion de base portant un budget dans son nom.

## Et le prix de l'alpha commun est tombé en prime

Sans une conversion de plus, sur les deux points :

```
budget 4,50 Gio :  45 groupes recuperables sur 64, sous 2 % de surcout relatif
budget 5,00 Gio :  40 groupes recuperables sur 64
```

À comparer aux **5 groupes sur 64** qui fusionnent aujourd'hui sur le tout-int8.
La couverture passerait donc de 7,8 % à ~70-78 %, et le gain de débit étant
proportionnel à la couverture — vérifié à deux ancrages, 6,5 % d'écart — cela
vaut environ **+1,7 à +1,9 point** de débit, pour zéro octet de plus.

**Ce que ce chiffre n'est pas :** le surcoût de 2 % est une erreur de **sortie
par tenseur**, pas une perplexité. Il dit que la reconversion vaut d'être
essayée ; il ne dit pas ce qu'elle coûterait à la perplexité. Le trancher
demande une conversion à `alpha` commun et un `acvram eval` contre 5,4141 — et
c'est mesurable, puisque les deux points ci-dessus donnent déjà la référence à
laquelle la comparer.

Note : **40 récupérables à 5,00 Gio contre 45 à 4,50** — moins de groupes
récupérables quand plus de tenseurs sont promus en int8. À expliquer avant d'en
tirer quoi que ce soit ; ce n'est pas encore un fait, c'est une observation sur
deux points.
