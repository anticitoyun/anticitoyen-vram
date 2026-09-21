# Q3N : d'où vient le bruit, et la table qui le supprime

Mesures du 8 septembre 2026, en float64 sur processeur, contre la source
réelle de la conversion. Aucune carte occupée.

## La source n'est pas ce qu'on croyait

Le manifeste du modèle converti porte `model.name = Qwen3-Coder-Next-heretic-Q3_K_S`,
et le seul dossier source présent sur les disques est ce GGUF **Q3_K_S**. La
conversion q3n part donc d'un modèle **déjà quantifié à environ 3,4 bits** : les
`out_snr_db` du manifeste mesurent un couple de grilles, pas la qualité des
poids. Il n'existe pas de poids en pleine précision à comparer ici, et toute
mesure présentée comme « contre l'original » serait fausse par sa légende.

## Un plancher, pas une distribution

Déciles du SNR sur les 73 872 tenseurs q3n :

    5,58 · 13,27 · 13,30 · 13,32 · 13,32 · 13,33 · 13,33 · 13,34 · 13,34 · 13,34 · 13,37

Quatre-vingt-dix pour cent des tenseurs tiennent dans **0,10 dB**. Des tenseurs
de tailles et de statistiques différentes ne donnent jamais cela : c'est un
plancher imposé par la table, pas une mesure du contenu.

Concentration des 68 tenseurs sous 10 dB : **67 sont des `down_proj` d'experts**,
un seul `gate_proj`, aucun `up_proj`.

## Deux mécanismes soupçonnés, un seul coupable

**Échelle FP8 écrasée ou tombée dans les subnormaux — RÉFUTÉ.** Les échelles de
bloc nulles existent (jusqu'à 44,4 % des blocs sur le pire tenseur) mais elles
recopient des blocs source **déjà vides** : la part d'énergie du tenseur qui
disparaît dans un bloc non nul rendu à zéro vaut **0,00 %** partout, et le SNR
calculé en les ignorant est identique à 0,01 dB près. Les blocs subnormaux
portent au plus 8,8 % de l'énergie et leur SNR propre n'est pas pire que celui
des blocs normaux. Un plancher d'échelle ne gagnerait rien de mesurable.

**Table sans niveau zéro — CONFIRMÉ, c'est tout l'écart.** Le plus petit niveau
de la table vaut 0,1025 × échelle du bloc ; un poids nul est donc reconstruit à
±0,1025 × amax de son bloc. Ce n'est pas une approximation dégradée, c'est de
l'**énergie créée**, et elle se compose couche après couche.

La variable qui sépare les deux populations est le creux du bloc — part des
poids sous un huitième de l'amax de leur bloc :

| population | creux | SNR |
|---|---|---|
| les vingt pires | 85 à 95 % | 5,58 à 9,07 dB |
| les tenseurs ordinaires | 21 % | 13,3 dB |
| blocs denses d'un tenseur ordinaire | — | 13,49 dB |
| blocs creux du **même** tenseur | — | 8,60 dB |

Et les tenseurs cassés sont des **experts quasi morts** : 97,4 %, 94,7 %, 89,9 %
et 90,1 % de zéros exacts pour les quatre pires, contre 20,7 % pour un expert
ordinaire. Le format dépense 3,25 bits par poids pour stocker du vide, et y
injecte du bruit.

## La table

Sept niveaux symétriques, un code sur huit inutilisé, **3,25 bits par poids
inchangés** :

    (-1 ; -0,5699 ; -0,2491 ; 0 ; 0,2491 ; 0,5699 ; 1)

Ajustée par Lloyd-Max sur 3 478 900 valeurs réduites tirées d'un échantillon
stratifié — 144 strates (48 couches × 3 projections), blocs à échelle nulle
écartés, symétrisation exacte à chaque itération, bornes fixées à ±1 pour
interdire tout écrêtage. Elle est symétrique, donc conforme à la spécification
du 8/09.

Gains mesurés sur **288 tenseurs disjoints de ceux de l'ajustement** :

| table | SNR médian | gain moyen | d1 | d5 | d9 | pire | % qui perdent |
|---|---|---|---|---|---|---|---|
| actuelle | 13,33 | — | — | — | — | — | — |
| zéro asymétrique | 14,27 | +0,99 | +0,94 | +0,94 | +0,97 | +0,93 | 0,0 |
| Lloyd sur les creux | 14,57 | +1,29 | +1,23 | +1,25 | +1,26 | +1,23 | 0,0 |
| **Lloyd stratifiée** | **15,01** | **+1,73** | +1,67 | +1,68 | +1,72 | **+1,65** | **0,0** |

**Aucun tenseur ne perd, sur 288.** Le pire cas est un gain. Les déciles tiennent
dans 0,05 dB : le gain est uniforme, ce n'est pas une moyenne qui cache une queue.
Sur les sept témoins choisis pour leur creux, la même table rend **5,58 → 22,20 dB**.

La table unique est donc justifiée : pas de table par tenseur, pas de bit
d'en-tête, pas de complexité à porter.

## Ce que ces mesures ne disent pas

- **Aucune perplexité.** Un SNR de poids ne prédit pas une sortie. +1,73 dB
  uniforme est une amélioration réelle du format ; ce qu'elle rend en perplexité
  reste à mesurer, contre l'étalon du modèle source.
- Le tirage aléatoire n'a attrapé qu'**un** tenseur au-delà de 60 % de creux sur
  288 : le +5,00 dB de cette catégorie repose sur un seul cas. Le gain massif sur
  les experts vides est établi séparément, par sept témoins choisis exprès.
- Les niveaux sont ajustés sur **ce** modèle. Pour un autre, ils sont à vérifier
  avant d'être repris — ce n'est pas une constante universelle du format.
- Le gain contre une source déjà quantifiée bénéficie en partie de la
  reproduction de ses zéros exacts. Contrôlé : en remplaçant chaque zéro de la
  source par un bruit du plus petit poids non nul, le gain sur les tenseurs
  cassés tient (+16,01 dB au lieu de +16,69). Sur du synthétique sans aucun zéro,
  le niveau zéro gagne 8,4 dB en queue lourde et 12,4 dB avec aberrants, mais
  **perd 0,8 dB** sur une gaussienne pure : le remède vaut pour des poids réels,
  pas pour n'importe quelle distribution.

## Étalons de perplexité établis le même jour

Corpus `wiki.test.raw` de wikitext-2-raw (miroir ggml-org), 1 290 590 octets,
sha256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
llama.cpp e34f042, `-c 512`, stride 512, 256 positions notées par fenêtre
(`perplexity.cpp:541`, `first = n_ctx/2`).

| modèle | fenêtres | perplexité |
|---|---|---|
| Qwen3-Coder-Next-heretic **Q3_K_S** (la source de q3n) | 584 | **9,1831 ± 0,073** |
| le même, cumul intermédiaire | 64 | 8,5236 |
| Qwen3.6-12B-IQ-Q5_K_M (témoin d'instrument) | 580 | **31,6919 ± 0,265** |
| phi-4-Q4_K_M (témoin dense) | 565 | **6,5988 ± 0,041** |
| le même, cumul intermédiaire | 64 | 29,5004 ± 0,721 |

Le nombre de fenêtres est propre au **tokeniseur du modèle**, jamais au corpus :
584 pour Coder-Next, 580 pour le 12B, 565 pour phi-4, sur le même fichier.
Comptes exacts par `llama-tokenize` : 297 193 jetons pour le 12B, 289 305 pour
phi-4. Un compte de fenêtres qui ne correspond pas est le contrôle de
tokenisation le plus court qui existe.

Trois règles qui se déduisent de ces chiffres :

1. **Une éval restreinte à N fenêtres se compare au cumul à la fenêtre N**, jamais
   au chiffre final. Le cumul oscille — 32,4 puis 26,2 puis 29,5 puis 31,7 sur le
   témoin — et viser le final imputerait au format jusqu'à 7 % de pur corpus.
2. **64 fenêtres ne suffisent pas à trancher un biais de 2 %** : l'incertitude y
   vaut ±2,4 %, plus grande que le seuil. Il faut le fichier entier, où elle
   tombe à ±0,84 %.
3. **Un chiffre ne se recopie pas sans son étiquette.** Trois erreurs le même
   jour, toutes de la même forme — une valeur mesurée juste, transportée avec une
   légende fausse : un compte d'octets lu dans un `ls` et écrit comme s'il avait
   été mesuré ; une cible établie sur 584 fenêtres appliquée à une mesure sur 64 ;
   un nombre de fenêtres transporté d'un modèle à l'autre, jusque dans le nom du
   fichier de journal. Aucune n'était une erreur de mesure, et c'est ce qui les
   rend dangereuses : le chiffre résiste à la relecture, sa légende non. La parade
   est la même que le sceau posé sur les tables q3n — **un chiffre voyage avec ce
   qui l'a produit** : modèle, corpus et son empreinte, découpage, nombre de
   fenêtres, incertitude. Une ligne de tableau sans ces champs n'est pas une
   mesure, c'est un souvenir.

Deux exécutions indépendantes rendent `[64] 29,5004` à la quatrième décimale :
llama.cpp est déterministe sur ce chemin, et l'incertitude publiée est bien de
l'échantillonnage du corpus, pas du bruit d'exécution.

## Comparer q3n à quoi

L'étalon 9,1831 vient de llama.cpp ; un chiffre q3n viendrait du harnais
acvram. **La comparaison serait inter-instruments**, et tout écart systématique
entre les deux implémentations s'imputerait au format sans que rien ne l'en
distingue. Il faut donc mesurer d'abord cet écart sur un modèle que les deux
outils servent, dans le même cadrage — le témoin 12B ci-dessus est là pour ça.
