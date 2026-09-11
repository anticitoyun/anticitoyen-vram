# Ce que la grille de `paged_attn_partial` devrait être

Chantier sans carte, demandé après l'établissement par convergence de la
sous-parallélisation (6,1× sa borne à ctx 924, 8,6× à ctx 3007). Tous les chiffres
ci-dessous sont **lus dans le binaire** `bbbb85c59e8f` ou dérivés d'eux, aucun n'est
supposé.

## 1. La capacité réelle, mesurée et non estimée

`cuobjdump --dump-resource-usage` sur l'instanciation qui nous concerne,
`paged_attn_partial<128, __nv_bfloat16, __nv_bfloat16>` :

    REG 40 par thread    SHARED 3 648 octets    128 threads (PA_WARPS=4)

Ce qui borne le nombre de blocs résidents par SM, sur 5090 (170 SM, 65 536 registres
et 48 warps par SM) :

    registres   65 536 / (40 x 128)  = 12 blocs
    warps       48 / 4               = 12 blocs      <- a egalite
    partagee    102 400 / 3 648      = 28 blocs
    plafond materiel                   32 blocs

**12 blocs par SM, et la limite est atteinte simultanément par les registres et par
les warps.** C'est un point de conception heureux : aucun des deux n'est à gaspiller,
et il ne sert à rien d'économiser des registres sans augmenter `PA_WARPS`.

Capacité de la carte : **2 040 blocs, 8 160 warps.**

## 2. Ce que nous lançons aujourd'hui

Grille `g1(BQ, HQ, C)` = `(1, 40, C)` avec `C = ceil(N x 16 / PA_CHUNK)`.
À ctx 1024 et `PA_CHUNK = 512` : `C = 2`, donc **80 blocs**.

    80 blocs sur 2 040          occupation 3,9 %
    80 blocs sur 170 SM         90 SM ne recoivent RIEN

Le second chiffre est le plus dur : **plus de la moitié de la carte ne travaille pas
du tout**, et aucune optimisation interne au noyau ne peut y changer quoi que ce soit.
C'est la forme du lancement qui l'interdit.

## 3. Les deux seuils qui comptent, et ils ne sont pas au même endroit

    tous les SM touches      >= 170 blocs   ->  C >=  5   ->  chunk <= 204
    occupation 50 %          >= 1 020 blocs ->  C >= 26   ->  chunk <=  40
    occupation 100 %         >= 2 040 blocs ->  C >= 51   ->  chunk <=  20

    chunk 512  ->    80 blocs    47 % des SM     3,9 % d'occupation
    chunk 256  ->   160 blocs    94 %            7,8 %
    chunk 205  ->   200 blocs   100 %            9,8 %
    chunk 128  ->   320 blocs   100 %           15,7 %
    chunk  64  ->   640 blocs   100 %           31,4 %
    chunk  32  ->  1 280 blocs  100 %           62,7 %
    chunk  16  ->  2 560 blocs  100 %          100,0 % (sature)

**Le premier seuil est le seul qui soit qualitatif** : sous 170 blocs, des SM sont
inertes ; au-dessus, tous travaillent et le reste est une affaire de latence cachée.
Il est franchi dès `chunk ≤ 204`, c'est-à-dire bien avant les valeurs que le balayage
a explorées.

Et la borne du code n'est pas la contrainte : `TORCH_CHECK(C <= 256)` autorise
10 240 blocs, **cinq fois la capacité utile**. Personne n'a à la relever.

## 4. Deux leviers, orthogonaux, et l'ordre entre eux n'est pas indifférent

- **`C` (le découpage en tranches)** ajoute des blocs, donc des SM. C'est le seul
  levier qui peut franchir le premier seuil. Il coûte : `C` fois plus de travail au
  `paged_attn_reduce`, un second lancement dès que `C > 1`, et `part`/`part_m`/`part_l`
  dimensionnés en `[BQ, HQ, C, D]` — la mémoire de travail croît linéairement.
- **`PA_WARPS`** ajoute des warps par bloc, donc du parallélisme **dans** un SM, sans
  toucher au `reduce`, ni au nombre de lancements, ni à la mémoire de travail. Coût :
  `sacc[PA_WARPS][D]` en mémoire partagée (8 Ko à 16 warps, encore loin des 100), et
  une réduction finale sérielle en `O(PA_WARPS)` dans le thread 0.

**L'erreur à ne pas faire est de croire que `PA_WARPS` remplace `C`** : 80 blocs de
16 warps font 1 280 warps mais laissent toujours **90 SM vides**. Le nombre de blocs
plafonne les SM utilisés, et rien d'autre ne le fait. L'ordre est donc :

    1. C assez grand pour depasser 170 blocs      (qualitatif, non negociable)
    2. PA_WARPS pour cacher la latence a l'interieur des SM ainsi actives
       — gratuit cote reduce, donc a preferer a un C encore plus grand

## 5. Ce qui reste à mesurer, et ce que chaque mesure trancherait

Rien de ce qui précède ne dit **où** s'arrête le gain, parce que trois coûts croissent
avec `C` et qu'aucun n'est chiffré :

    a) le reduce : son temps en fonction de C, a mesurer seul
    b) la memoire de travail : C x BQ x HQ x D flottants, allouee par appel
    c) le second lancement : deja compte, 1,1 us par frontiere

Le balayage existant montre un plateau à 11,5 µs de 64 à 768 blocs puis 15,36 µs à
1 504 — donc **le coût d'occupation apparaît vers 768 blocs**, soit un peu avant le
seuil des 50 %. C'est cohérent avec la capacité calculée ici : à 768 blocs on a
4,5 blocs par SM, et le SM commence à arbitrer entre eux.

À contexte court, le premier seuil devient inatteignable par `C` : à ctx 128, même
`chunk = 32` ne donne que 160 blocs, sous 170. **C'est le seul régime où `PA_WARPS`
est le levier unique**, et c'est aussi celui du décodage à contexte frais — donc il
n'est pas marginal.

## 6. Ce que je ne prétends pas

Aucune de ces lignes n'est une mesure de temps. Elles disent ce que la carte **peut**
héberger et ce que nous lui **donnons**, deux comptes exacts ; elles ne disent pas de
combien le noyau accélérera. Le gain de +88/+104 % déjà obtenu porte sur `C` seul ;
l'effet de `PA_WARPS` n'a jamais été mesuré et pourrait être nul si la latence est
déjà cachée par les 4 warps existants.

---

# `PA_WARPS` paramétrable : le patch, le protocole, et la prédiction

## 1. Un défaut du contrôle d'empreinte, découvert en préparant ce patch

`_SRC_HASH` était calculé sur le **seul contenu du `.cu`**. Or un paramètre passé par
`-D` ne change pas le fichier : **quatre valeurs auraient donné le même
`ACVRAM_SRC_HASH`**, donc `ccache` pouvait rendre le même objet, donc quatre réglages
auraient rendu quatre fois le même chiffre. C'est mot pour mot le défaut que le
commentaire du contrôle décrit comme fermé — et `ACVRAM_GW_WARPS` y était **déjà**
exposé, `PA_WARPS` l'aurait été.

Corrigé : le hash porte sur le couple `(contenu, flags qui varient)`. Le contrôle
d'empreinte devient alors ce qu'il prétendait être, et il distingue deux binaires qui
ne diffèrent que par un `-D`.

## 2. Le patch

`PA_WARPS` est paramétrable **à la compilation** et non à l'exécution, sans choix
possible : `sacc[PA_WARPS][D]` est une déclaration de mémoire partagée, la valeur doit
être connue de `nvcc`. D'où `-DPA_WARPS=n`, avec deux `static_assert` :

    1 <= PA_WARPS <= 32       32 warps = 1024 threads, le plafond d'un bloc
    puissance de 2            la repartition t += PA_WARPS reste reguliere

Et une validation Python **hors du `try`** de `get_extension()`. Ce point n'est pas
cosmétique : à l'intérieur, une `ValueError` aurait été avalée par le
`except Exception` et rendue comme *« repli sur les noyaux de référence »* — une
valeur hors domaine aurait pris l'apparence d'une extension indisponible. **Un refus
doit refuser.** Vérifié sans compiler :

    ACVRAM_PA_WARPS=9      LEVE
    ACVRAM_PA_WARPS=zero   LEVE
    ACVRAM_PA_WARPS=4      -DPA_WARPS=4
    ACVRAM_PA_WARPS=16     -DPA_WARPS=16

## 3. Ce que `PA_WARPS` change, et ce qu'il ne change pas

    PA_WARPS   threads   blocs/SM   warps/SM   a 80 blocs : warps actifs
           4       128         12         48        320    ( 3,9 % de 8160)
           8       256          6         48        640    ( 7,8 %)
          16       512          3         48       1280    (15,7 %)
          32      1024          1         32       2560    (31,4 %)

**Le fait qui gouverne tout : `warps/SM` vaut 48 dans les trois premiers cas.** À SM
saturé, `PA_WARPS` est donc rigoureusement **neutre** — il ne fait que répartir les
mêmes warps dans moins de blocs. Il ne paie **que** parce que nous sommes à 80 blocs
sur 2 040, c'est-à-dire loin de la saturation : chaque SM actif reçoit 0 ou 1 bloc, et
augmenter `PA_WARPS` remplit ce bloc unique.

Corollaire à retenir : **le jour où `C` remplira la carte, `PA_WARPS` cessera de
payer.** Les deux leviers ne s'additionnent pas indéfiniment ; le second n'a de valeur
que dans le régime où le premier ne peut rien.

## 4. Le protocole — `PA_WARPS` se mesure à `C` constant

Sinon les deux leviers se mélangent et on ne saura pas lequel a payé.

    PA_CHUNK fixe a 512 dans TOUS les bras (donc C constant a contexte donne)
    PA_WARPS parcourt 4, 8, 16, 32 — un binaire par valeur, empreinte publiee
    deux contextes :  ctx 128  (C = 1, le regime ou C ne peut rien)
                      ctx 1024 (C = 2, celui du tableau)
    ordre ABBA sur chaque paire, dispersion publiee, temps en ms brutes
    controle du binaire : ACVRAM_PA_WARPS=9 doit LEVER avant toute mesure
    controle du montage : le pas median a PA_WARPS=4 doit retrouver la valeur
                          deja mesuree (33,286 ms a ctx 1024) — c'est le temoin
                          que seul le parametre a change

**Le point de mesure qui compte est ctx 128, pas 3007.** À ctx 128 avec `chunk = 512`,
`C = 1` : **40 blocs seulement, 130 SM vides**, et aucune valeur de `PA_CHUNK` ne peut
y remédier puisque le contexte lui-même borne `C`. C'est le régime du décodage à
contexte frais — le début de toute conversation.

## 5. Prédiction, écrite avant la mesure

À **ctx 1024** : gain réel mais borné, parce que 90 SM restent vides quoi qu'on fasse
aux warps. J'attends que le temps de l'attention (5,198 ms mesurés à `PA_WARPS=4`)
descende de **20 à 45 %** entre 4 et 16, et que 32 n'apporte rien de plus que 16 —
voire régresse, l'occupation par SM y retombant de 48 à 32 warps.

À **ctx 128** : c'est là que j'attends le plus, en proportion, puisque `C = 1` prive
la carte de tout autre levier. Mais je n'ai **aucune mesure du temps de l'attention à
ctx 128**, donc je prédis un sens et non une valeur : décroissance monotone de 4 à 16,
et le maximum du gain relatif de toute la campagne.

**Ce qui me réfuterait** : un temps plat de 4 à 32. Cela voudrait dire que 4 warps
cachent déjà toute la latence, et alors le noyau serait limité par autre chose que le
parallélisme — ce qui contredirait les 6,1× et 8,6× établis par convergence, et
rouvrirait la question au lieu de la fermer. Je l'ai écrit dans le document initial et
je le maintiens : **l'effet de `PA_WARPS` pourrait être nul.**

---

# RECTIFICATION — ma prédiction sur `PA_CHUNK` est réfutée, et mon candidat d'explication aussi

## 1. Ce que j'avais écrit, ce que la mesure dit

J'attendais 50 % d'occupation à `chunk ≤ 40` et 100 % à `chunk ≤ 20`, et j'en tirais
qu'il restait du gain sous 64. **L'optimum mesuré de sept valeurs est 64**, et le
travail cesse de décroître en dessous : 10,30 µs à 64, 10,31 à 32 — identiques au
centième, alors que la proportionnalité était vérifiée de 128 à 2048.

Mon calcul comptait ce que la carte **peut héberger**. Il ne comptait pas si le
travail découpé **suffit à remplir** un bloc. Sous un certain seuil, ajouter des blocs
ne remplit plus la carte : **ça vide les blocs**. Le mécanisme d'occupation était le
bon, son domaine de validité ne l'était pas — et un modèle juste sur la forme et faux
sur son domaine rend la bonne réponse pour la mauvaise raison.

## 2. Un candidat d'explication, écarté par son propre calcul

J'ai d'abord soupçonné le trafic de `part` / `part_m` / `part_l`, que mon document ne
comptait pas et qui croît **linéairement avec `C`** : `[BQ, HQ, C, D]` en float32,
écrit par `partial` puis relu par `reduce`.

    chunk    C   par couche    par pas   a 1050 Go/s
      512    2        41 Ko     4,0 Mo     0,004 ms
       64   16       328 Ko    32,2 Mo     0,031 ms
       32   32       655 Ko    64,4 Mo     0,061 ms
       16   64      1310 Ko   128,8 Mo     0,123 ms

À `chunk = 16` ce trafic dépasse les 92,3 Mo de cache KV lu par pas — le découpage fin
fait plus de trafic que la donnée qu'il traite. **Mais il ne peut pas être la cause** :
0,123 ms, soit 2,4 % des 5,198 ms de l'attention, et même à 6× de sa borne comme le
KV il ne pèserait que 0,74 ms. **Écarté en deux minutes par le calcul, au lieu d'une
nuit de mesures** — c'est le seul intérêt de l'avoir chiffré.

## 3. Ce qui reste sans cause, dit comme tel

Le plancher double entre `chunk 64` et `chunk 16` (15,36 → 33,73 µs). Le seul
mécanisme compatible avec le balayage du matin est que **le plancher croît avec le
nombre de blocs au-delà de ~768**, ce que ce balayage montrait déjà (plat à 11,5 µs de
64 à 768 blocs, puis 15,36 à 1 504). La cause de cette croissance est **inconnue**, et
je n'ai pas de candidat qui survive à son propre calcul. Elle est à publier
quantifiée, pas distribuée.

## 4. Ce que ça change pour `PA_WARPS` — et une correction à ce qu'on en conclut

Il a été dit que `C` ne remplira jamais la carte, donc que `PA_WARPS` reste le seul
levier. C'est vrai sur le fond, mais **le chiffre ne se transporte pas** : les 1 504
blocs à `chunk 64` sont ceux d'un autre modèle. Chez nous, à ctx 1024 et `HQ = 40` :

    chunk 512  ->   80 blocs    3,9 % de la capacite
    chunk 128  ->  320 blocs   15,7 %
    chunk  64  ->  640 blocs   31,4 %      <- l'optimum mesure
    chunk  32  -> 1280 blocs   62,7 %

**À l'optimum de découpage, nous n'utilisons que 31 % de la capacité.** Mon corollaire
— `PA_WARPS` cesse de payer quand `C` remplit la carte — ne mord donc pas encore : il
reste les deux tiers de la carte à remplir par les warps, et c'est vrai **même à
ctx 1024**, pas seulement à ctx 128.

Ce qui renforce le protocole au lieu de l'affaiblir, et déplace une chose : le point de
mesure principal reste ctx 128 (`C = 1`, 40 blocs, 130 SM vides, aucun découpage n'y
peut rien), mais ctx 1024 n'est plus un simple témoin — à 31 % d'occupation, il doit
lui aussi montrer un gain. **S'il n'en montre pas alors que ctx 128 en montre, c'est
mon modèle d'occupation qui tombe une seconde fois**, et il faudra alors chercher ce
qui borne le noyau ailleurs que dans le parallélisme.

---

# RESULTAT `PA_WARPS` — 10/09 10:42, six points, trois empreintes distinctes

    PA_WARPS   ctx    pas (ms)   p (ms)   p / pas    contre PA_WARPS=4
           4   128      28,687    0,612    2,13 %          —
           8   128      28,279    0,367    1,30 %      -40,0 %
          16   128      28,134    0,185    0,66 %      -69,7 %
           4  1024      33,176    5,198   15,67 %          —
           8  1024      30,620    2,678    8,74 %      -48,5 %
          16  1024      29,355    1,409    4,80 %      -72,9 %

Contrôles passés aux six points : bras inconnu qui lève, trois sorties distinctes,
chaque bras reproductible (0,005 à 0,025 ms entre occurrences), graphes actifs,
0 couche exilée, **trois empreintes de `.so` distinctes**. Reproductibilité entre
campagnes : le pas à `PA_WARPS=4` ctx 1024 vaut 33,176 ms contre 33,286 ce matin,
soit 0,3 %.

## 1. Le gain, et il est moteur, pas seulement de noyau

    ctx  128 : 28,687 -> 28,134 ms   -1,9 % de temps par jeton, +2,0 % de debit
    ctx 1024 : 33,176 -> 29,355 ms  -11,5 % de temps par jeton, +13,0 % de debit

**Un paramètre de compilation, une ligne, +13 % de débit à ctx 1024.**

## 2. Mon modèle d'occupation tient — et ma prédiction est fausse deux fois

La clause était écrite : *si ctx 1024 ne montre aucun gain quand ctx 128 en montre,
mon modèle d'occupation tombe une seconde fois.* Les deux montrent du gain, monotone,
dans le sens prédit. **Le modèle tient.** Mais les deux chiffres que j'avais avancés
sont faux :

- **l'amplitude** : j'annonçais −20 à −45 % entre 4 et 16, c'est **−72,9 %**. Sous-
  estimé d'un facteur 1,6 à 3,6.
- **le classement** : j'annonçais le maximum du gain relatif à ctx 128 « puisque `C`
  n'y peut rien ». Le maximum est à **ctx 1024** (−72,9 contre −69,7 %). Mon
  raisonnement confondait *« `C` ne peut rien »* avec *« `PA_WARPS` peut plus »* :
  l'effet de `PA_WARPS` ne dépend pas de `C`, il est donc du même ordre partout, et
  c'est là où l'attention **pèse** le plus qu'il rapporte le plus.

## 3. Le fait le plus important : le seuil est franchi dans l'autre sens

À `PA_WARPS = 16` et ctx 1024, `p` tombe à **4,80 % du pas mesuré, sous le seuil de
5,24 %**. L'attention cesse donc d'être le poste qui contredit la conclusion
d'`ETABLI.md:2476`. Autrement dit : **cette conclusion n'était pas fausse en soi, elle
était fausse pour le noyau d'alors**, et un correctif de compilation la rend vraie.
C'est le premier cas de la journée où un seuil est franchi par une optimisation et non
par une correction de mesure.

## 4. Ce que ça change pour `PA_WARPS = 32`, contre ma propre prédiction

J'avais écrit que 32 n'apporterait rien, voire régresserait, l'occupation par SM y
retombant de 48 à 32 warps. **La décroissance observée est monotone et loin de
s'aplatir** : −40 %, puis −69,7 % ; rien n'annonce un retournement à 16. Le point 32
redevient donc intéressant, et il est bloqué par une instanciation `D = 512` **que
personne n'utilise** (0 modèle sur 120 recensés). **Le plafonnement des warps par
instanciation passe de « à faire si utile » à prioritaire.**

## 5. Une coïncidence à ne pas prendre pour une validation

À `PA_WARPS = 16`, mon pas vaut 29,355 ms contre 28,748 attendus au banc — écart
+2,2 %, dans les « quelques pourcents » que mon contrôle exigeait. **Il ne faut pas en
conclure que mon montage est devenu conforme.** Le banc de référence tournait à
`PA_WARPS = 4`, réglage où mon montage donne 33,176 ms, soit +15,4 %. Mes conditions
n'ont pas changé entre les deux réglages : l'accord numérique à 16 est **fortuit**, et
le montage reste 15 % plus lent que le banc à réglage égal. Le conclure serait la même
faute que le dénominateur emprunté — prendre un chiffre qui s'approche pour un chiffre
qui correspond.

## 6. Réserve sur le transport

Tout ceci porte sur `paged_attn_partial`, donc sur les modèles à attention paginée.
`mla.py` ne contient **aucun** appel à ce noyau : ces gains ne se transportent pas aux
modèles MLA du parc, GLM-4.7 compris.
