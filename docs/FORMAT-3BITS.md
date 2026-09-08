# Un format sous quatre bits pour les experts

Spécification écrite le 8 septembre 2026. **Aucun noyau n'est écrit** : ce
document dit quoi écrire et pourquoi ce choix plutôt qu'un autre. Les chiffres
de qualité sont **recalculés sur processeur**, pas repris d'une publication.

## Pourquoi

Un modèle trois bits reste aujourd'hui non convertible : la conversion le ferait
grossir, ce que la garde refuse désormais. Le refus est juste et ne débloque
rien. Il faut un format qui tienne réellement sous quatre bits par poids.

## Ce qui a été mesuré

Rapport signal sur bruit, un million de poids, deux distributions : gaussienne,
et à queue lourde — celle qu'ont les couches de mélange d'experts, où quelques
poids valent beaucoup plus que la médiane. La queue lourde est une gaussienne
modulée par une lognormale de **paramètre σ = 0,5**. Il faut le dire, parce que
c'est une queue *modérée* et que le résultat en dépend fortement : voir plus
bas. Échelle stockée en FP8 e4m3, arrondie
comme elle le sera réellement.

Toutes les mesures passent par le **vrai chemin** : l'échelle est arrondie en
FP8 avant de quantifier, comme elle le sera à l'exécution. Arrondir après, comme
le faisait une première version de ce calcul, flatte le résultat d'environ un
décibel.

| Format | bits/poids | gaussien | queue lourde |
|---|---|---|---|
| entiers 3 bits, bloc 32 | 3,25 | 12,90 dB | 10,77 dB |
| entiers 3 bits, bloc 16 | 3,50 | 14,02 dB | 12,43 dB |
| **quantiles 3 bits, bloc 32** | **3,25** | **15,00 dB** | **13,57 dB** |
| **quantiles 3 bits, bloc 16** | **3,50** | **15,85 dB** | **15,05 dB** |
| entiers 4 bits, bloc 32 | 4,25 | 20,21 dB | 17,79 dB |
| entiers 2 bits, bloc 16 | 2,50 | 4,66 dB | 4,86 dB |

**Le choix se lit dans ce tableau.** Les quantiles à 3,25 bits battent les
entiers à 3,50 — moins de mémoire *et* meilleure qualité, sur les deux
distributions. Sur la queue lourde, l'écart atteint 2,6 dB à taille égale, parce
que des niveaux également espacés gaspillent leur résolution là où il n'y a
presque rien.

Deux bits sont hors de question : 4,7 dB, c'est un bruit du même ordre que le
signal.

### Un piège écarté en chemin : la table doit être symétrique

Une première table reprenait la forme de NF4 — quatre niveaux négatifs, un zéro
exact, trois positifs, le plus grand positif valant 0,675. Avec une échelle
prise sur le maximum **absolu**, tout poids positif proche du maximum était
écrêté d'un tiers.

Le défaut s'est vu à une anomalie et non au raisonnement : **le bloc de 32
donnait un meilleur rapport que le bloc de 16**, ce qui est impossible — un bloc
plus grand partage une échelle entre plus de poids et ne peut que perdre. La
table symétrique rétablit l'ordre attendu et gagne 1,7 dB en gaussien, 2,6 sur
la queue lourde.

### Combien la queue pèse

Le chiffre annoncé pour la queue lourde vaut pour σ = 0,5, et rien au-delà. En
faisant varier σ sur la même graine :

| σ | bloc 32 | bloc 16 | écart |
|---|---|---|---|
| 0 (gaussien) | 14,99 dB | 15,84 dB | 0,85 dB |
| **0,5 (le chiffre de ce document)** | **13,55 dB** | **15,03 dB** | **1,48 dB** |
| 1,0 | 9,49 dB | 12,06 dB | 2,57 dB |
| 1,5 | 6,67 dB | 9,58 dB | 2,91 dB |

**Deux choses en découlent, et elles vont dans le même sens.** Le format perd
vite quand la queue s'alourdit : à σ = 1,0 on est à 9,5 dB, c'est-à-dire dans
le régime où le bruit commence à compter. Et l'écart entre les deux tailles de
bloc se creuse au lieu de se refermer — 0,85 dB en gaussien, 2,91 dB à σ = 1,5.

Autrement dit, **plus les experts réels ont une queue lourde, plus le bloc de
16 devient le bon choix**, et moins le quart de bit économisé se défend. Une
mesure de la lourdeur de queue sur de vrais poids d'experts tranche donc
l'arbitrage sans qu'il faille attendre une perplexité complète : c'est le
premier chiffre à prendre.

### Ce que les vrais poids ont répondu

Mesuré le 8 septembre sur les experts de Coder-Next, 60 tenseurs de la couche
12, par `outils/mesurer-queue-experts.py` :

| | bloc 32 | bloc 16 | écart |
|---|---|---|---|
| poids réels | 13,65 dB | 15,02 dB | **1,36 dB** |

**σ inter-blocs mesuré : 0,03. La queue des experts est quasi gaussienne.**
C'est le contraire de ce que la prudence suggérait, et cela tranche l'arbitrage
dans le sens du bloc de 32 : à σ ≈ 0 le tableau ci-dessus annonce 0,85 dB, on
en mesure 1,36, et non les 2,5 à 3 dB qui auraient justifié le quart de bit.

Deux précautions sur ce chiffre, toutes deux importantes.

**σ ne se mesure pas sur les poids individuels de cette source.** Une première
estimation, par `Var(log|w|) − π²/8`, rendait zéro partout — en contradiction
avec un SNR qui correspondait à σ ≈ 0,5. Elle avait tort : la source est déjà
en NVFP4, ses poids ne prennent que huit amplitudes par bloc et un cinquième
vaut exactement zéro. Cette variance mesurait la grille du format source, pas
la queue du modèle. L'amplitude maximale par bloc, portée par l'échelle FP8,
survit à la quantification : c'est sur elle que σ se lit, et c'est aussi ce que
q3n voit.

**Ce SNR compare deux grilles de quantification, pas q3n aux poids d'origine.**
Il n'existe aucune source bf16 de ce modèle sur le disque ; la conversion q3n
part du NVFP4, donc quantifie une seconde fois. L'écart *entre tailles de bloc*
reste lisible — les deux subissent la même source — mais aucune conclusion sur
la **table** ne peut venir de là. Une table à zéro exact y gagne 3,8 dB, ce qui
n'est qu'un alignement sur la grille source : sur des poids continus elle en
perd 1,3, et 21 % de zéros injectés dans un gaussien n'en rendent que 0,4.

## Le format retenu

**Quantiles normaux 3 bits, échelle FP8 e4m3 par bloc, blocs de 32. 3,25 bits
par poids.** L'arbitrage, laissé ouvert dans une première version, est tranché
par la mesure ci-dessus : la queue des experts réels est quasi gaussienne et
l'écart vaut 1,36 dB. Le bloc de 32 reste le bon choix.

Le bloc de 16 coûte 0,25 bit de plus et rend 0,85 dB en gaussien, **1,48 dB sur
la queue lourde** — et c'est la queue lourde qui décrit les experts. Une table
symétrique a resserré l'écart en gaussien mais l'a creusé sur la distribution
qui compte.

Le bloc de 32 reste le défaut parce que ce format existe pour faire tenir un
modèle qui ne tient pas : 3,25 contre 3,50 bits, c'est 7 % de mémoire en moins
sur les experts, et c'est la raison d'être de tout l'exercice. Mais 1,48 dB
n'est pas un dixième de décibel qu'on écarte d'un mot : **les deux blocs sont
spécifiés à égalité** — même code, une constante — et le choix appartient à une
mesure de perplexité sur un vrai modèle, pas à ce document.

### Les huit niveaux

Quantiles de la demi-normale aux milieux des huitièmes, normalisés, puis
**symétrisés** — quatre niveaux positifs et leurs opposés :

```
[-1.0000, -0.5783, -0.3186, -0.1025, 0.1025, 0.3186, 0.5783, 1.0000]
```

**Il n'y a pas de zéro exact**, et c'est délibéré : avec huit niveaux, dépenser
l'un d'eux sur une valeur exacte coûte plus qu'il ne rapporte, un poids nul
étant représenté à 0,10 échelle près sans que cela change le produit. Le
symétriser importe davantage.

Ils sont **figés dans le code**, pas recalculés : deux implémentations qui les
calculent séparément divergeraient au dernier bit, et un poids quantifié
ailleurs ne se relirait plus.

### Empaquetage

Trois bits ne s'alignent pas sur l'octet. **Huit valeurs tiennent exactement sur
trois octets**, et c'est le seul groupement qui tombe juste sous 32 bits.

Un bloc de 32 poids fait donc 4 groupes de 3 octets, soit **12 octets**, plus
1 octet d'échelle : 13 octets pour 32 poids, soit 3,25 bits par poids.

Dans un groupe de 8 valeurs `v0..v7`, chacune sur 3 bits, l'ordre est du poids
faible vers le poids fort du mot de 24 bits :

```
mot = v0 | v1<<3 | v2<<6 | v3<<9 | v4<<12 | v5<<15 | v6<<18 | v7<<21
octet0 = mot & 0xFF ; octet1 = (mot>>8) & 0xFF ; octet2 = (mot>>16) & 0xFF
```

**Ce choix rend la déquantification lisible sans table** : trois octets chargés
d'un coup en 32 bits, huit décalages, huit masques à 7. Une disposition qui
couperait une valeur entre deux octets obligerait à recoller à travers la
frontière, pour rien.

### Disposition en mémoire

Trois tenseurs, comme le format quatre bits existant, pour que le compte des
transferts par poids reste juste :

- `qweight` : `uint8`, `[sortie, entrée*3//8]` — les index empaquetés ;
- `block_scale` : `float8_e4m3fn`, `[sortie, entrée//32]` ;
- `global_scale` : `float32`, scalaire.

`entrée` doit être un multiple de 32. Le remplissage se fait comme aujourd'hui,
et une couche qui ne s'y plie pas n'est pas convertie plutôt que d'être
convertie de travers.

## Le chemin de calcul

**Pas d'instruction native, et il n'en faut pas.** La déquantification vers bf16
se fait en mémoire partagée, comme le fait déjà la multiplication groupée : on
déballe une tuile de poids, on l'écrit en bf16 partagé, on multiplie. Le coût
est un déballage par tuile, amorti sur toute la tuile.

Les huit niveaux tiennent dans **une table de 8 valeurs bf16 en mémoire
constante**, lue par tous les fils. Un index de 3 bits y accède directement :
pas de calcul, une lecture.

## Les trois fonctions à écrire

```python
def quantize_q3n(w: torch.Tensor, block: int = 32) -> Q3NTensor:
    """Quantifie [sortie, entrée] vers quantiles 3 bits, échelle par bloc.

    L'échelle d'un bloc est max(|w|) du bloc, arrondie au FP8 e4m3 le plus
    proche AVANT de quantifier les poids — sinon la quantification vise une
    échelle que le stockage ne saura pas rendre, et l'erreur d'arrondi de
    l'échelle s'ajoute à celle des niveaux au lieu d'être absorbée.
    """

def dequantize_q3n(t: Q3NTensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """Reconstruit [sortie, entrée]. Chemin de référence, hors noyau."""

def q3n_gemv(x: torch.Tensor, t: Q3NTensor) -> torch.Tensor:
    """``x @ W.T`` pour un jeton, déquantification par tuile en partagé.

    x : [1, entrée] ou [entrée]. Sortie : [1, sortie].
    Renvoie None si la forme ne convient pas — même contrat que les autres
    chemins, pour que le repli reste possible sans exception.
    """
```

## Ce que cette spécification ne dit pas

- **La perplexité.** Le rapport signal sur bruit n'est pas la qualité perçue :
  15 dB sur les poids ne dit pas ce que le modèle répondra. La comparaison à
  faire est perplexité contre Q3_K_S sur le même texte, et elle demande une
  carte.
- **Le débit du noyau.** Un format qui divise la mémoire par deux et le débit
  par trois ne sert à rien. À mesurer contre le chemin quatre bits existant.
- **Le comportement des experts réels.** Les distributions testées sont
  synthétiques, et le tableau de sensibilité ci-dessus montre qu'un σ mal deviné
  déplace le résultat de plusieurs décibels. C'est la première chose à refaire
  quand un modèle trois bits sera sous la main.
