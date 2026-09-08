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

## 8 septembre 2026, soir — la table quitte la spécification pour le manifeste

L'hypothèse « à huit niveaux, la symétrie vaut plus qu'un zéro » est réfutée
par la mesure (session de mesure, 8/09) : sur 288 tenseurs d'évaluation
disjoints des 288 d'ajustement, sept niveaux symétriques AVEC zéro gagnent
+1,73 dB en moyenne, zéro perdant, déciles dans 0,05 dB — et une table à
huit niveaux symétriques SANS zéro, même réajustée au mieux, reste loin
derrière : c'était le zéro qui manquait, pas le placement. Sur les blocs
creux (85-95 % des poids sous un huitième de l'amax du bloc), le plus petit
niveau non nul reconstruisait chaque poids nul à ±0,1025 × amax — de
l'énergie créée, qui se compose couche après couche.

Ce qui change :

* **La table vit dans le manifeste**, par modèle (champ `table` de chaque
  entrée q3n, huit flottants). Les niveaux s'ajustent par modèle — Lloyd-Max
  sur un échantillon stratifié (couches × projections), évalué sur un
  échantillon disjoint. `TABLE_Q3N` de la spécification d'origine reste le
  repli des manifestes qui ne déclarent pas la leur.
* **Invariants structurels**, vérifiés à toute lecture (`valider_table_q3n`) :
  huit entrées physiques (le masque & 7 du dépaquetage exige un index
  toujours valide — une table à sept niveaux répète le dernier), croissance,
  bornes ±1 exactes (aucun écrêtage), symétrie. La symétrie est un invariant
  DE LA SPÉCIFICATION, pas de la structure : une table asymétrique avec zéro
  (quatre négatifs, zéro, trois positifs) est à l'étude — l'argument est que
  la règle de symétrie visait l'écrêtage de NF4 et que ±1 présent des deux
  côtés l'écarte — et son adoption relâcherait ce seul contrôle.
* **Un sceau lie la table aux poids** : sha256 des 64 premiers octets du
  qweight et de la table, dans chaque entrée, vérifié au chargement. Un
  manifeste régénéré sans reconversion refuse de servir.
* **Le noyau reçoit la table en argument** et la lit depuis la mémoire
  partagée — la version `__constant__` se sérialisait dès que les fils d'un
  warp lisaient des entrées différentes, c'est-à-dire toujours.

Questions ouvertes, à trancher par la mesure : les VALEURS définitives des
niveaux (passe stratifiée à trois tables en cours — sept symétriques avec
zéro, huit libres avec zéro, huit symétriques sans zéro) ; l'asymétrie
(arbitrage de l'auteur de la spécification) ; et le fait qu'un code sur huit
inutilisé est de la capacité laissée sur la table — la mesure dit que ce
gaspillage rapporte quand même, pas qu'aucune table à huit niveaux ne ferait
mieux. Un SNR de poids ne prédit pas une perplexité : la décision finale
appartient à l'éval avant/après sur le même cadrage.

### Arbitrage du même soir — la symétrie tombe, l'invariant reste

L'auteur de la spécification relâche la symétrie : la règle d'origine
interdisait l'asymétrie alors qu'elle voulait interdire l'ÉCRÊTAGE, et ce
sont les bornes ±1 aux deux extrémités qui l'empêchent. Vérification faite
avant l'accord, pas sur parole : le biais d'une table asymétrique (erreur
non centrée, qui se somme linéairement sur K là où un bruit croît en
racine de K, invisible dans un SNR de tenseur) existe — 260 fois celui de
la table d'origine — mais ne se matérialise pas en sortie ; sur un
down_proj alimenté par ses vraies activations, la part d'erreur portée par
la moyenne vaut 0,002 pour la table à huit niveaux avec zéro contre 0,010
pour la table d'origine. L'invariant du validateur : huit entrées, bornes
±1 exactes, croissance strictement monotone (égalité admise entre les deux
dernières entrées seulement), au plus un zéro.

Et la mise en garde qui justifie la table PAR MODÈLE, mesurée sur tenseur
dense gaussien : les tables à zéro y PERDENT (13,62 à 14,52 dB contre
15,00 pour la table d'origine). Le « zéro perdant sur 288 » est une
propriété des tenseurs de CE modèle — creux ou à queue lourde —, pas des
tables à zéro. **Une table q3n vaut pour la distribution sur laquelle elle
a été ajustée, et pour aucune autre** ; graver ces niveaux comme constante
universelle serait l'erreur inverse de celle que la première spécification
a commise.
