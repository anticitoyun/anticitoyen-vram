# La courbe du quota en six points — et le défaut est dans le sac à dos, pas dans le quota

```
point                   bits/p      PPL   vs etalon   promus   bits/PPL   alpha rec.
plancher tout-nvfp4     4,7297   5,6102    +3,622 %       —          —          —
budget 4,50 Gio         5,7396   5,5643    +2,774 %  110/225      22,00      45/64
budget 5,00 Gio         6,3737   5,5394    +2,314 %  147/225      25,47      40/64
budget 5,50 Gio         6,9905   5,5125    +1,817 %  172/225      22,93          —
budget 6,00 Gio         7,6320   5,4918    +1,435 %  198/225      30,99      38/64
plafond tout-int8       8,3453   5,4144    +0,006 %       —       9,22       5/64
```

Étalon extérieur `transformers`/GPTQ : **5,4141**, corpus `wiki-gptq.txt`
(sha `e52922746ad09bac`). Dispersion de l'instrument : **0,000000 PPL** à six
décimales, remesurée sur le binaire courant.

## Monotone, sans plateau — l'hypothèse de départ reste réfutée

Aucun palier nulle part. Le coût en octets d'un point de perplexité **monte**
dans l'intérieur de la courbe — 22,00 puis 25,47 puis 22,93 puis 30,99 — donc
il n'y a **aucun octet gratuit à reprendre en bas**, contrairement à ce que le
feu vert du matin supposait.

## Mais le dernier segment vaut 9,22, soit 3,4 fois mieux que celui d'avant

Et c'est le vrai résultat. Décomposé par tenseur :

```
172 -> 198 promus   26 tenseurs   0,6415 bits   0,0207 PPL   30,99 bits/PPL   0,796 milli-PPL/tenseur
198 -> 225 promus   27 tenseurs   0,7133 bits   0,0774 PPL    9,22 bits/PPL   2,867 milli-PPL/tenseur
```

**Les 27 tenseurs que le sac à dos refuse en dernier rendent 3,7 fois plus de
perplexité par tenseur que les 26 qu'il accepte juste avant.** L'ordre du
glouton est donc inversé sur la queue.

La cause est dans son critère. `convert.py:1068` ordonne par

```python
ordre = sorted(budget_candidats, key=lambda c: -c["gain_db"] / c["cout"])
```

c'est-à-dire par **gain de SNR par octet**. Or ce qui décide est la perplexité
par octet, et les deux ne coïncident pas : les tenseurs à faible gain de SNR par
octet portent ici l'essentiel de la perplexité. Le sac à dos est optimal — pour
un objectif qui n'est pas le nôtre.

**Ce qui manque pour l'affirmer, et je ne l'affirme donc pas encore :** le
plafond n'est pas un point de budget mais un changement de format, produit par
le mécanisme de plancher SNR et non par le sac à dos. Les 27 tenseurs qu'il
promeut ne sont peut-être pas exactement ceux que le glouton a refusés.

**Le témoin plafond de poste4 tranche exactement cela** : une conversion à
`--bits-budget 6.55` passe par le sac à dos et promeut tout, ou presque. Si elle
retrouve 5,4144, la comparaison est homogène et l'inversion d'ordre est établie.
Si elle trouve nettement plus haut, alors c'est le mécanisme qui diffère et non
l'ordre. **La réserve posée ce matin sur les témoins refaits se paie ici, sur un
résultat qu'elle est seule à pouvoir valider.**

## Et l'alpha récupérable décroît avec le nombre de promus

```
110/225 promus  ->  45/64 recuperables sous 2 %
147/225         ->  40/64
198/225         ->  38/64
225/225         ->   5/64 qui fusionnent DEJA (grandeur differente)
```

Trois points monotones : **plus il y a de tenseurs promus en int8, moins il y a
de groupes où un exposant commun est bon marché.** L'observation que je donnais
sur deux points tient sur trois. Je m'abstiens d'un mécanisme : la dernière
ligne ne mesure pas la même chose que les trois autres (« fusionnent déjà »
contre « récupérables »), et je n'ai pas de troisième route pour la vérifier.

Ce que cela change pour le chantier `alpha` : le gain est **plus grand sur les
dossiers les moins promus** — donc sur les modèles serrés en mémoire, ceux où le
débit compte le plus. C'est favorable, et c'est l'inverse de ce que j'aurais
supposé.
