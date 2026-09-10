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

**Dans l'unité qui décide, et c'est une rectification** : un sac à dos ordonne
par gain **par octet**, pas par tenseur. J'avais publié « facteur seize » en
milli-PPL par *tenseur* — grandeur qui invite exactement la question que
personne n'avait posée : et si les derniers tenseurs étaient simplement plus
gros ? Relevé par chef, converti, et l'inversion tient — mais le facteur
global vaut **5,4 et non 16**.

```
segment              n    bits   milli-PPL   /tenseur   bits/tens   milli-PPL PAR BIT
  0 ->   6  (poste4) 6  0,0551      1,10      0,183     0,00918          19,96
172 -> 198  (poste1) 26  0,6415     20,70      0,796     0,02467          32,27
198 -> 225  (poste1) 27  0,7133     77,40      2,867     0,02642         108,51

facteur sur toute la course, par TENSEUR    15,64x   <- ce que j'avais publie
facteur sur toute la course, par BIT         5,44x   <- l'unite qui decide
dernier contre precedent, par BIT            3,36x
```

**La réserve que personne n'avait posée est donc levée** : les 27 derniers sont
**1,071 fois plus gros** et rendent **3,6 fois plus** de perplexité. Si le
rapport de tailles avait valu 3,6 au lieu de 1,07, il n'y aurait eu aucune
inversion et le résultat serait tombé. Il tient — dans la bonne unité, et de
moins loin que ce que mon premier chiffre laissait croire.

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

## Et l'inversion se teste directement, sans témoin

Proposition de chef, meilleure que mon attente : à **budget constant**, même
sac à dos, même plancher, même format, deux bras qui ne diffèrent que par le
**signe** de la clé de tri.

```
budget 6,00 Gio
  bras A   ordre = -gain_db / cout     l'ordre actuel
  bras B   ordre = +gain_db / cout     le meme code, un signe
```

Si le critère est bien orienté, **B doit être nettement pire que A**. S'il est
mal orienté sur la queue, B sera meilleur ou équivalent — et ce serait la
démonstration, sans qu'aucune comparaison de **mécanisme** n'intervienne. Le
contrôle ne peut donc pas confirmer l'hypothèse par construction, ce qui est
exactement ce qui manquait à ma comparaison avec le plafond.

L'écart A−B donne au passage la **borne** de ce que l'ordre vaut, quel que soit
l'ordre optimal. Et l'ordre inverse n'est pas un candidat : c'est un instrument.

`ACVRAM_ORDRE_SAC_INVERSE` renverse le signe et rien d'autre, le sens est
inscrit au manifeste (`budget.ordre_glouton`) pour qu'un dossier produit par le
bras inverse soit reconnaissable sans son journal, et une épreuve CPU vérifie
que l'échappement renverse bien l'ordre au lieu de le permuter autrement.

**Le témoin plafond de poste4 reste utile pour autre chose** : il tranche : une conversion à
`--bits-budget 6.55` passe par le sac à dos et promeut tout, ou presque. Si elle
retrouve 5,4144, la comparaison est homogène et l'inversion d'ordre est établie.
Si elle trouve nettement plus haut, alors c'est le mécanisme qui diffère et non
l'ordre. **La réserve posée ce matin sur les témoins refaits se paie ici, sur un
résultat qu'elle est seule à pouvoir valider.**

## Et l'alpha récupérable décroît avec le nombre de promus

**Deux colonnes, pas une** — remarque de chef, et elle est juste : une colonne
unique fait croire à une série là où les deux dernières lignes ne mesurent pas
la même grandeur.

```
promus      recuperables sous 2 %   fusionnent deja
110/225            45/64                   —
147/225            40/64                   —
198/225            38/64                   —
225/225 (tout-int8)   —                   5/64
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

## Le témoin plafond ferme la réserve de mécanisme

Mesuré par poste4, et les 128 octets d'écart qu'elle attribuait au bruit sont
maintenant **nommés** :

```
                       fichiers        en-tetes      donnees        tenseurs
tout-int8 (format)   7 029 266 352     116 656   7 029 149 696        996
plafond (sac a dos)  7 029 266 480     116 784   7 029 149 696        996
ecart                       +128           +128            0            0
```

**Les octets de données sont identiques à l'octet**, et les 996 clefs de
tenseurs sont les mêmes. Les 128 octets vivent entièrement dans les en-têtes
safetensors, et leur cause est un **découpage de fragments différent** — 600/396
tenseurs contre 639/357 — qui change la longueur des décalages écrits en JSON :

```
int8      70 584 + 46 056 + 16 o de prefixes = 116 656
plafond   75 008 + 41 760 + 16               = 116 784
```

Aucun `__metadata__` d'aucun côté. **Ce n'est donc pas « probablement du
bruit » : c'est une frontière de fragment, et le contenu est le même.**

**Conséquence : ma seconde hypothèse tombe.** Le sac à dos à budget égal au
plafond et la conversion par format produisent le **même dossier**. Les deux
mécanismes convergent quand le budget les y force, donc mon plafond de référence
et un point de budget ne diffèrent pas par leur mécanisme. Si la perplexité
confirme 5,4144, la courbe est homogène sur toute sa longueur et **l'inversion
d'ordre du glouton est établie sans réserve de mécanisme**.
