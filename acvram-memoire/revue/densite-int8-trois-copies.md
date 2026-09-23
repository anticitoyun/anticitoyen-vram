# Densité int8 : l'incohérence venait de mon chiffre, pas du code — mais il y avait bien un défaut

## Ce que Jérôme a demandé
Lire la taille de groupe réelle de notre int8 et refaire `8 + 16/groupe`. Son
soupçon : `int8 = 9,500` surestime de 12 %, et un budget surestimé déclenche un
exil inutile, à 61-70 % du débit.

## Le contrôle

`formats.py:83-85` — la densité est déjà paramétrée par le groupe :

```python
if self.name == "int8":
    g = group_size or 128
    return 8.0 + 16.0 / g + 8.0 / g      # échelle fp16 + zéro uint8
```

La taille de groupe de production est **128** (`ConversionOptions.group_size = 128`,
`cli.py:654 --group-size default=128`). Donc :

```
int8 g=128   8,18750      <- le regime de production
int8 g= 64   8,37500
int8 g= 32   8,75000
int8 g= 16   9,50000      <- le chiffre que j'ai publie
int4_awq g=128  4,15625
int4_awq g= 16  5,25000   <- l'autre chiffre que j'ai publie
```

**Les deux chiffres que j'avais relayés sont les densités analytiques évaluées à
groupe 16 — la taille de bloc du NVFP4 — et non à la taille de production 128.**
Quatrième faute de relais de la journée, de la seconde espèce nommée par Manon :
chiffre exact, mauvais régime. Il n'y avait rien à vérifier dans le chiffre,
seulement le régime auquel je l'appliquais.

## L'incohérence se dissout, et elle se retourne en contrôle positif

Avec la vraie valeur, le mesuré repasse **au-dessus** du théorique, comme
attendu :

```
format    theorique   mesure disque   exces     part 16 bits deduite
nvfp4       4,5000        4,733       +0,2330       2,026 %
int8        8,1875        8,349       +0,1615       2,067 %
```

L'excès s'écrit `f × (16 − bpw)` où `f` est la part de poids gardée en 16 bits
(`embed_tokens`, normes, conv1d…). Les deux formats, quantifiés séparément et
mesurés séparément, **déduisent la même part à 2 % près : 2,026 % et 2,067 %**,
soit ~137 M poids sur 6,7384e9. Deux mesures indépendantes qui tombent sur la
même constante inobservée : le meilleur contrôle positif qu'on pouvait espérer,
et il valide `formats.py` autant que les deux conversions.

## Mais son instinct trouvait bien quelque chose : une TROISIEME copie

`convert.py:170` portait un dictionnaire écrit en dur — la troisième copie de la
même grandeur après `FormatSpec.bpw` et `FormatSpec.bits_per_weight()` — et
c'est **elle** qui alimente `bpw_cible`, donc le budget mémoire :

```
format      BPW_NOMINAL (convert.py)   formats.py g=128   ecart
int8               8,250                   8,18750        +0,76 %   ( +50,2 Mio)
int4_awq           4,250                   4,15625        +2,26 %   ( +75,3 Mio)
```

À groupe 128 l'écart est petit **et dans le sens prudent** (surestime le coût,
donc ne peut pas faire croire qu'un modèle tient) : aucun exil injustifié n'a pu
en venir. Le vrai danger était ailleurs : **la constante ignorait
`--group-size`**. À groupe 32, l'int8 réel vaut 8,750 et la constante 8,250
**sous-estimait de 5,71 %, soit 401 Mio** — le sens qui fait croire qu'un modèle
tient là où il ne tient pas, et qui se paie en OOM ou en exil de secours.

## Le correctif

`BPW_NOMINAL` supprimé. `convert.py` a maintenant

```python
def bpw_nominal(fmt: str, group_size: int = 128) -> float:
    try:
        return formats.bits_per_weight(fmt, group_size=group_size)
    except KeyError:
        return 16.0
```

et les quatre sites d'usage passent `opts.group_size` : `bpw_cible` (deux fois,
dont la bascule q3n) et `cout_promotion_mib`. Source unique : `quant/formats.py`.

Vérifié : `bpw_nominal` rend 8,1875 / 4,15625 à g=128 et 8,750 / 4,625 à g=32 ;
un format inconnu rend toujours 16,0 comme avant ; `pytest -k "promotion or cout"`
→ 3 passés.

## Second geste demandé : le renommage

`evaluate.py` — le champ `bits_per_weight` devient
**`bits_par_poids_en_memoire`**, avec en commentaire ce qui le rend indicatif :
il mesure `nbytes/params` après chargement, donc il dépend du dtype (26,99 en
fp32 contre 20,04 en bf16 pour un même dossier à 16,00 bits sur disque) et il
inclut caches et tampons. La clef JSON suit. Le seul consommateur était un test,
mis à jour.

## Ce qui reste

Le contrôle `model.nbytes` juste après le chargement des poids, avant toute
allocation de cache ou de tampon, pour nommer le facteur résiduel 1,30.

---

# Le compte ferme exactement, et ma déduction était elle-même un dénominateur emprunté

Jérôme demandait de remplacer la déduction cohérente par un compte qui ferme.
Les manifestes et les en-têtes safetensors suffisent — aucun chargement.

## Chaque octet des trois dossiers est attribué

```
Llama-2-7b-fp16pur   13 476 872 072 o   ecart au fichier +0
  weight       291 x  13 476 831 232   16,0000 b/p
  inv_freq      32 x           4 096    0,0000
  en-tetes                    36 744

Llama-2-7b-int8       7 029 266 352 o   ecart au fichier +0
  qweight      225 x   6 607 077 376    7,8441 b/p
  weight        66 x     262 676 480    0,3119
  scales       225 x     103 235 584    0,1226
  zeros        225 x      51 617 792    0,0613
  act_scale    223 x       4 538 368    0,0054   <- hors densite analytique
  inv_freq      32 x           4 096    0,0000
  en-tetes                   116 656

Llama-2-7b-nvfp4      3 983 834 740 o   ecart au fichier +0
  qweight      225 x   3 303 538 688    3,9220 b/p
  block_scale  225 x     412 942 336    0,4903
  weight        66 x     262 676 480    0,3119
  act_scale    224 x       4 554 752    0,0054   <- hors densite analytique
  global_scale 225 x             900    0,0000
  inv_freq      32 x           4 096    0,0000
  en-tetes                   117 488
```

`P = 6 738 417 664` poids comptés dans le manifeste, part 16 bits **1,9491 %**
= 131 338 240 poids = embedding 32 000 × 4 096 + 65 normes × 4 096. **Exactement
le compte à la main de Jérôme**, au poids près.

Et sur les seuls safetensors, le fp16 pur donne **16,0000 bits/poids** — pas
16,0006 : mon 16,00 était mesuré sur le dossier entier, tokenizer compris.

## Ma déduction était fausse en méthode, et la bonne forme donne l'égalité exacte

J'écrivais `f = excès / (16 − bpw)` en comparant le mesuré à `bpw` appliqué à
**tous** les poids. Or la densité analytique ne vaut que pour les 98,051 % qui
sont quantifiés. C'était un dénominateur emprunté de plus — dans mon propre
contrôle. La forme juste :

```
excès = mesuré − bpw × 0,98051
int8    8,3452 − 8,1875 x 0,98051 = 8,3452 - 8,0279 = 0,3174
nvfp4   4,7296 − 4,5000 x 0,98051 = 4,7296 - 4,4123 = 0,3174
```

**Identiques à quatre décimales**, et non « pareils à 2 % près ». Et l'excès se
décompose sans reste :

```
part 16 bits  1,9491 % x 16          = 0,31186
act_scale                            = 0,00539
inv_freq + en-tetes                  = 0,00014
                                       0,31739
```

Ce que mes 2,026 % et 2,067 % mesuraient réellement, c'était la même grandeur
vue à travers deux dénominateurs différents : l'écart de 2 % entre mes deux
déductions n'était pas du bruit de mesure, c'était l'erreur de méthode.

## `model.nbytes` : la valeur prévue, pour que le relevé puisse se tromper

`model.nbytes` = `embed_tokens.numel × element_size(dtype de chargement)` +
Σ `QuantLinear.nbytes`. Il exclut donc trois choses, toutes chiffrées ci-dessus :
les **65 normes** (pas des `QuantLinear`), les **`act_scale`** (dans le
`ChannelScaler`, pas dans `qweight.nbytes`) et `inv_freq`. Total exclu 0,0060 b/p.
Et il **promeut** l'embedding — et, pour un modèle non quantifié, tous les poids
— au dtype de chargement.

```
modele       chargement bf16    chargement fp32    fichier
fp16pur        15,9994            31,9987          16,0000
int8            8,3391             8,6504           8,3452
nvfp4           4,7235             5,0347           4,7296
```

**Conséquence directe : le couple 26,99 (fp32) / 20,04 (bf16) que j'ai publié ne
peut pas venir de `Llama-2-7b-fp16pur`**, dont la prévision est 31,9987 / 15,9994.
Je ne sais pas de quel modèle il venait, et je ne l'attribue donc pas — le
nommer au hasard serait la faute de la journée une cinquième fois. Ce qu'on peut
dire de lui sans son dossier : en écrivant `nbytes = A + B·s` où `B` est la part
promue, `26,99 − 20,04 = 6,95 b/p` se double, soit **43 % des poids restés en
`PlainTensor`** — ce n'est pas un dossier purement quantifié. Et `26,99 / 20,04
= 1,3467` : le « facteur résiduel 1,30 » était le rapport fp32/bf16 d'un même
relevé, pas un écart au disque.

Le relevé de Jérôme (`nbytes` juste après le chargement, avant tout cache) garde
tout son sens, mais il n'a plus besoin d'être exploratoire : il a maintenant une
**valeur prévue** pour chacun des trois étalons, donc il peut rendre « faux » de
façon détectable. Il demande un chargement, donc l'accord des sessions.
