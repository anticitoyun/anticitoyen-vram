# Étalons de perplexité et biais d'instrument

Mesures du 8 septembre 2026. Tout ce qui suit a servi à trancher un dossier de
format ; ce document existe pour qu'aucun de ces chiffres ne soit recopié sans
ce qui l'a produit.

## Le corpus, une fois pour toutes

`wiki.test.raw` de wikitext-2-raw, miroir ggml-org sur Hugging Face — l'archive
que télécharge `scripts/get-wikitext-2.sh` de llama.cpp ; le miroir S3
historique ne répond plus.

    1 290 590 octets · 241 211 mots
    sha256 173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08

## Ce que llama-perplexity mesure exactement

`tools/perplexity/perplexity.cpp:541` : `const int first = n_ctx/2`. Le
programme découpe le fichier en fenêtres **disjointes** de `n_ctx` jetons et ne
note que la **seconde moitié** de chacune. À `-c 512` : 256 positions notées par
fenêtre, chacune ayant au moins 256 jetons de contexte devant elle.

Un harnais tiers n'est comparable que s'il note les mêmes positions :
`fenêtre 512, stride 512, contexte minimal 256`. Pas 128, pas « contexte long ».

## Étalons

llama.cpp e34f042, `-ngl 999`, RTX 5090.

| modèle | n_ctx | fenêtres | perplexité |
|---|---|---|---|
| Qwen3-Coder-Next-heretic **Q3_K_S** (source de q3n) | 512 | 584 | **9,1831 ± 0,0730** |
| Qwen3.6-12B-IQ-Q5_K_M (hybride) | 128 | 2321 | 52,5769 ± 0,4729 |
| | 512 | 580 | **31,6919 ± 0,2652** |
| | 2048 | 145 | **24,0799 ± 0,1918** |
| phi-4-Q4_K_M (dense) | 128 | 2260 | 9,1320 ± 0,0651 |
| | 512 | 565 | **6,5988 ± 0,0414** |
| | 2048 | 141 | 5,8406 ± 0,0353 |
| phi-4 requantifié **Q8_0** depuis le Q4_K_M | 512 | 565 | 6,5974 ± 0,0414 |

Comptes exacts de jetons (`llama-tokenize`) : 297 193 pour le 12B, 289 305 pour
phi-4, 299 182 pour Coder-Next (par son `tokenizer.json`).

## Trois règles que ces chiffres imposent

**1. Le nombre de fenêtres appartient au tokeniseur du modèle, pas au corpus.**
584 · 580 · 565 sur le même fichier. C'est le contrôle de tokenisation le plus
court qui existe : un harnais qui annonce un autre compte a un problème
d'encodage, et aucun format n'est en cause.

**2. Une éval restreinte à N fenêtres se compare au cumul à la fenêtre N**,
jamais au chiffre final. Le cumul de Coder-Next passe par 8,52 à la fenêtre 64
pour finir à 9,18 ; celui du 12B par 29,50 pour finir à 31,69. Viser le chiffre
final imputerait au format jusqu'à 7 % qui n'est que du corpus.

**3. La taille d'échantillon doit être choisie d'après le seuil qu'on veut
trancher.** À 64 fenêtres l'incertitude vaut ±2,4 %, plus grande que le seuil de
2 % qu'on s'était donné : le test ne pouvait pas répondre à sa propre question.
Sur le fichier entier elle tombe à ±0,84 %.

Deux exécutions indépendantes rendent `[64] 29,5004` à la quatrième décimale :
llama.cpp est déterministe sur ce chemin, l'incertitude publiée est bien de
l'échantillonnage du corpus.

**Dette de preuve, à mon débit.** Ce chiffre est consigné ici et dans
`VALIDATION-FORMAT-Q3N.md` au moment de la mesure, avec son incertitude — ce
n'est pas un souvenir. Mais **les journaux des deux exécutions ne sont plus
produisibles** : ils étaient dans le worktree `/tmp`, qui ne survit pas à un
redémarrage. La conclusion que ce chiffre porte est la plus structurante du
dossier — c'est elle qui sépare la qualité, intacte, de la performance, à
refaire à chaque changement de régime. Elle repose donc sur une **trace
secondaire**.

**SOLDÉE le 8/09 à 21h51.** Relance à l'identique — même build `e34f042`, même
corpus (1 290 590 o, sha `173c87a5…`), mêmes conditions — après redémarrage et
dans un état de machine franchement différent :

    [64] 29.5004     Final estimate: PPL = 29.5004 +/- 0.72123

Reproduit à la quatrième décimale, incertitude comprise. Journal conservé **hors
de `/tmp`** : `acvram-memoire/journaux/etalon-29-5004.log`. L'insensibilité au
régime n'est plus seulement consignée, elle est **reproduite**.

## Le biais d'instrument, et pourquoi il fallait le mesurer

Comparer un chiffre produit par acvram à un étalon produit par llama.cpp est une
comparaison **inter-instruments**. Un écart systématique entre les deux
implémentations s'imputerait intégralement au format sans que rien ne l'en
distingue — et il ne se manifesterait pas comme une ambiguïté, mais comme un
écart net, crédible et faux. C'est le cas dangereux.

Mesuré sur phi-4, dense, format à 44 dB, sans récurrence :

| fenêtre | llama.cpp | acvram int8 | écart |
|---|---|---|---|
| 128 | 9,1320 | 9,592 | +5,0 % |
| 512 | 6,5988 | 6,911 | +4,7 % |
| 2048 | 5,8406 | 5,984 | +2,45 % |

La part revenant à la requantification est **nulle** : le même modèle
requantifié en Q8_0 et mesuré par llama.cpp rend 6,5974 contre 6,5988, vingt
fois moins que l'incertitude. La totalité de l'écart est un biais d'instrument.

**Il n'est pas constant** — il décroît avec la fenêtre. Il ne doit donc jamais
être soustrait d'un chiffre mesuré à une autre fenêtre que celle où il a été
établi : la correction aurait l'air rigoureuse et fabriquerait une erreur.

## Ce que l'étalonnage a trouvé

Sur le témoin **hybride**, l'écart valait ×2,16 à 128 jetons, ×4,08 à 512, ×9,21
à 2048 — et la perplexité acvram **empirait** avec le contexte (113, 129, 222)
quand la référence s'améliorait (53, 32, 24). Un modèle qui prédit moins bien
avec plus d'information n'a pas une récurrence imprécise : il en a une qui
injecte de l'information fausse.

Cause trouvée dans les métadonnées, pas dans un profil : le GGUF porte
`qwen35.rope.dimension_sections = [11, 11, 10, 0]`, c'est-à-dire un M-RoPE
**entrelacé**, là où un RoPE NEOX standard était appliqué. L'affectation
fréquence→paire est permutée, les angles relatifs sont faux, l'erreur croît avec
l'écart de position et n'atteint que les couches d'attention pleine. Les trois
signatures observées s'expliquent d'un coup.

Vérification faite sur l'autre modèle plutôt que déduite de la table
d'architectures : `Qwen3-Coder-Next-heretic-Q3_K_S` déclare `qwen3next`,
`rope.freq_base` 5 000 000, `rope.dimension_count` 64, et **aucun
`dimension_sections`**. Un NEOX standard y est correct.

Un RoPE peut encore être faux de trois façons qu'aucune relecture de
`rotate_half` ne montre : la base theta, le nombre de dimensions tournées,
l'origine des positions. Les deux premières se lisent dans les métadonnées en
deux minutes. C'est ce qui a nommé le défaut ici, avant qu'une comparaison
d'activations couche par couche ne soit lancée.

## La leçon commune

Trois défauts trouvés le même jour — un corpus de 283 jetons noté depuis la
position zéro, une architecture servie avec le mauvais type de RoPE, un modèle
converti sans filet de promotion pendant que son comparant en avait un — ont la
même forme : **le défaut n'était pas le calcul, mais le silence du calcul sur ce
qu'il ne savait pas faire.** Un outil qui rend un nombre là où il devrait
refuser coûte plus cher qu'un outil qui tombe en panne.

---

# Bilan : avant / après, et ce qui reste non mesuré

Table tenue à jour au fil des mesures. **Une case vide est marquée « non
mesuré », jamais laissée absente** : une ligne manquante se lit comme un oubli,
une case marquée se lit comme un travail à faire.

## Les étalons de référence (llama.cpp, tous mesurés)

| modèle | rôle | n_ctx | fenêtres | perplexité |
|---|---|---|---|---|
| Qwen3-Coder-Next **Q3_K_S** | source de q3n, **la cible** | 512 | 584 | 9,1831 ± 0,0730 |
| Qwen3.6-12B Q5_K_M | témoin hybride | 128 | 2321 | 52,5769 ± 0,4729 |
| | | 512 | 580 | 31,6919 ± 0,2652 |
| | | 2048 | 145 | 24,0799 ± 0,1918 |
| phi-4 Q4_K_M | témoin dense | 128 | 2260 | 9,1320 ± 0,0651 |
| | | 512 | 565 | 6,5988 ± 0,0414 |
| | | 2048 | 141 | 5,8406 ± 0,0353 |
| phi-4 **Q8_0** requantifié | coût d'une requantification 8 bits | 512 | 565 | 6,5974 ± 0,0414 |

Conditions communes : llama.cpp e34f042, `wiki.test.raw`
sha256 `173c87a5…7dd08`, stride = n_ctx, n_ctx/2 positions notées par fenêtre,
`-ngl 999`, RTX 5090.

**Ces étalons sont insensibles au régime de puissance, au mode persistant et au
démarrage à froid.** Une perplexité est une fonction déterministe des poids et
des jetons : aucune grandeur temporelle n'y entre. Le premier passage d'une
mesure peut coûter une seconde de plus, cela ne change ni la log-vraisemblance
d'une position ni sa moyenne.
Preuve, et non affirmation : deux exécutions indépendantes du même modèle, à des
heures différentes et dans des états de machine différents, l'une avec
`--chunks 64` et l'autre sans, rendent **`[64] 29,5004` à la quatrième
décimale**. Un coût de démarrage qui entrerait dans le chiffre les aurait fait
diverger.
**Conséquence pratique** : ces chiffres restent comparables à toute mesure
future, quel que soit le bridage des cartes ou l'état du mode persistant. C'est
ce qui sépare définitivement le dossier **qualité**, intact, du dossier
**performance**, à refaire à chaque changement de régime — où il faut au
contraire jeter le premier passage, et **dire lequel des deux coûts on jette**,
puisqu'un banc qui exclut le temps de première réponse du débit cache la moitié
du surcoût sans le signaler.

## Le format q3n, avant / après

| état | perplexité | rapport à 9,1831 | conditions |
|---|---|---|---|
| q3n **sans filet** (`snr_floor` 0, aucune promotion) | 1005,838 | ×110 | 584 fenêtres, 148 920 positions, contexte minimal 256 |
| q3n **avec filet** (`snr_floor` 25) | **209,036** | ×22,8 | idem, 144 tenseurs promus en int8 |
| **NVFP4** (4,5 bits), mêmes conditions | **193,621** | ×21,1 | idem — le témoin qui rend les autres lisibles |
| q3n avec filet **et** table Lloyd | *non mesuré* | — | ne peut agir que sur les 0,077 nats propres au format |

### Décomposition, en nats — c'est elle qui conclut

| poste | nats | lecture |
|---|---|---|
| gain du filet (`snr_floor` 0 → 25) | **−1,571** | facteur 4,81 sur la perplexité. Le filet est le résultat principal du dossier côté format. |
| écart de format restant, q3n − NVFP4 | **+0,077** | +8,0 % relatif, pour **1,25 bit de moins par poids** |
| excès commun aux deux formats | **+3,049** | facteur 21,1 sur l'étalon. **97,5 % de l'écart total.** |

**Ce que ces trois lignes établissent, et qui renverse le dossier.**

1. **Le filet était le défaut principal.** Une conversion sans plancher de SNR
   laissait 143 `shared_expert` — sur le chemin de 100 % des jetons, à chacune
   des 48 couches — en 3,25 bits là où le NVFP4 les protégeait en int8. Le
   corriger vaut un facteur 4,8. Ce n'était pas un réglage, c'était le sujet.

2. **Le format q3n tient sa promesse.** À 3,25 bits contre 4,5, il lit 27,8 %
   d'octets en moins et coûte **8 % de perplexité**. L'incertitude
   d'échantillonnage sur 148 920 positions vaut environ 0,5 %, donc l'écart est
   réel — mais il est petit, et c'est la bonne nouvelle : le format n'est pas le
   problème.

3. **Le poste dominant n'est pas le format, il est au moteur.** L'excès commun
   aux deux formats — 3,05 nats, facteur 21 — ne peut venir ni de q3n ni de
   NVFP4 puisqu'il leur est commun. C'est le « fait sans cause » des
   architectures hybrides, mesuré cette fois sur `qwen3next` et non plus
   soupçonné sur un témoin. **Il pèse 97,5 % de l'écart à l'étalon.**

Et un fait qui oriente sa recherche : le NVFP4 **génère du texte cohérent en
usage réel** avec une perplexité de harnais de 193. Une perplexité de 193
signifierait un modèle inutilisable ; le modèle ne l'est pas. Le défaut est donc
probablement dans le chemin d'**évaluation ou de préremplissage** sur les
hybrides, et non dans la génération.

**Aucun de ces chiffres n'est une perplexité du format q3n**, et ils ne doivent
pas être écrits ainsi. Ils sont mesurés par le harnais acvram ; l'étalon vient
de llama.cpp. Deux réserves les grèvent, et elles sont de nature différente :

**1. Le biais d'instrument, mesuré, non constant.** Sur un modèle dense, format
à 44 dB, sans récurrence : +5,0 % à 128, +4,7 % à 512, +2,45 % à 2048. La part
revenant à la requantification est nulle (le Q8_0 ci-dessus le prouve). Ce biais
ne doit jamais être soustrait d'un chiffre mesuré à une autre fenêtre que celle
où il a été établi.

**2. Un fait sans cause, sur les architectures hybrides.** Sur le témoin 12B,
les sorties de couche d'acvram s'écartent de celles de llama.cpp de façon
croissante avec la profondeur — cosinus 0,997 à la couche 0, 0,660 à la couche
18 — et la dégradation se concentre entre couches **GDN**, les couches
d'attention pleine n'étant pas touchées. Mais la même couche prise **isolément**
est saine, avec un écart plat. Le fait est donc établi en assemblage et sans
cause identifiée. Rien ne prouve à ce jour que `qwen3next` — l'architecture de
Coder-Next, hybride elle aussi — le partage ; rien ne prouve le contraire.

Conséquence pratique : **la distance d'un chiffre q3n à 9,1831 n'est pas
attribuable au format** tant que ce fait n'est pas expliqué. La **différence
entre deux formats mesurés dans les mêmes conditions** l'est, elle, parce
qu'un défaut moteur commun s'y annule. C'est pourquoi le NVFP4 figure dans la
table : il n'est pas un chiffre de plus, c'est ce qui rend les autres lisibles.

## L'hypersensibilité des architectures hybrides : mesurée, et insuffisante

Hypothèse posée pour expliquer les 3,05 nats sans supposer aucun défaut : une
récurrence peut être **exacte** et **hypersensible** à la fois — un juge qui
prouve « mêmes poids → même sortie » ne prouve jamais « poids légèrement
différents → sortie proche ». Testée en requantifiant les deux témoins par la
même chaîne, et en mesurant avec le même instrument.

| format | hybride 12B | dégradation | phi-4 dense | dégradation | rapport |
|---|---|---|---|---|---|
| étalon | 31,6919 | — | 6,5988 | — | — |
| Q4_K_S | 32,6334 | **+2,97 %** | 6,6311 | **+0,49 %** | **6,1×** |
| Q3_K_S | 35,4686 | +11,92 % | 7,0475 | +6,80 % | 1,8× |

**L'hypersensibilité est réelle et chiffrée pour la première fois** : à
perturbation modérée, l'architecture hybride est **six fois** plus sensible que
la dense. C'est un résultat en soi.

**Elle n'explique pas l'écart.** La pire requantification que llama.cpp sache
infliger à ce modèle coûte +11,9 % ; l'int8 d'acvram, à 44 dB, est une
perturbation **plus fine** et coûte +308 %. Il manque un facteur **26**. Une
perturbation plus petite ne peut pas produire un effet vingt-six fois plus
grand — et l'effet **sature** quand la perturbation grossit (le rapport tombe de
6,1 à 1,8), au lieu de s'emballer comme il le faudrait.

Deux conséquences :
- **les hybrides se requantifient** : +11,9 % pour passer à 3,4 bits est un coût
  normal. La conclusion « il faut partir des poids d'origine en bf16 » tombe ;
- **l'excès de 3,05 nats reste sans cause**, et toutes les explications par le
  format ou par la perturbation des poids sont désormais éliminées par la mesure
  — le filet (réparé), le format q3n (8 %), la sensibilité architecturale (12 %
  au pire). Ce qui reste est au moteur.

Réserves à garder avec ces chiffres : requantifier depuis un Q5_K_M est une
double quantification, donc ces dégradations sont **majorées** — ce qui joue en
faveur de l'hypothèse, pourtant réfutée. Et les grilles de llama.cpp ne
dégradent pas exactement les mêmes tenseurs qu'un int8 de groupe 128.

Ce test chiffre au passage une confusion de ce document : l'étalon 9,1831 porte
sur les poids **d'origine**, quand les mesures acvram portent sur des
**requantifications**. La confusion existe donc bien — et elle vaut au plus 12 %.

## Ce qu'il reste à mesurer, par ordre de ce que ça rapporte

1. **Trancher évaluation contre génération.** Mesurer la perplexité en mode
   génération — jeton par jeton avec le cache, comme en usage réel — au lieu du
   préremplissage par fenêtres. Si elle s'effondre vers l'étalon, le défaut est
   dans le chemin d'éval et les 3,05 nats ne concernent pas le service. Si elle
   reste haute, le service est bien touché et c'est la priorité du projet. Un
   seul essai départage, et il conditionne tout le reste.
2. Le biais d'instrument sur architecture hybride, non borné à ce jour : les
   +4,7 % mesurés valent sur un modèle dense.
3. La table Lloyd, qui ne peut agir que sur les 0,077 nats propres au format.

## Sur la table Lloyd, si elle est un jour retenue

Son gain mesuré — +3,43 dB en moyenne sur 288 tenseurs disjoints, aucun tenseur
perdant — vaut pour **requantifier un modèle déjà quantifié avec une grille
contenant un zéro**. C'est le cas de Coder-Next, dont la source est un GGUF
Q3_K_S. **Elle est inutile sur un modèle bf16 : son gain vient des zéros exacts
de la grille source, pas d'une propriété du format.** Mesuré sur
DeepSeek-Coder-V2-Lite en bf16 d'origine, où la table de la spécification gagne :
14,79 contre 14,22 dB.

Et un décibel de SNR de poids ne prédit aucune perplexité. C'est la mesure
avant/après, sur le même corpus et dans le même cadrage, qui décidera — ou qui
dira que le gain ne se voit pas en sortie, ce qui devra être écrit aussi
franchement que le gain lui-même.

---

# Résolution : d'où viennent les 3 nats

Le dossier ouvert le matin — « q3n dégrade le modèle » — se referme le soir sur
une cause qui n'a rien à voir avec le format.

## La référence au grain de la position

Les logits complets de llama.cpp sur les 512 premiers jetons, toutes positions
(par défaut le graphe n'élabore que la dernière : il faut demander les logits à
chaque position du lot). Contrôle avant usage : perplexité recalculée depuis ces
logits **5,0467** contre **5,0411** rendu par `llama-perplexity` sur la même
fenêtre — 0,1 % d'écart.

| poste, positions 256-511 | llama.cpp | acvram |
|---|---|---|
| NLL médiane | **0,258** | 3,58 |
| NLL moyenne | 1,619 | 4,684 |
| part portée par les 10 % pires | **48,4 %** | 29,6 % |
| top-1 | **63,2 %** | 32,2 % |
| corrélation NLL / identifiant du jeton | 0,163 | 0,434 |

## Deux lectures renversées par ces chiffres

**La queue n'était pas le symptôme — elle était plus légère du côté malade.**
48,4 % du coût sur un dixième des positions chez le moteur sain, contre 29,6 %.
Un moteur sain *concentre* son coût sur les positions réellement difficiles ;
celui-ci l'étale. Chercher des accidents localisés était donc chercher à
l'envers.

**Le défaut frappe les positions FACILES.** Médiane 0,258 contre 3,58 : un
facteur quatorze là où la référence est quasi certaine. Ventilé par quartile de
difficulté de la référence, l'écart vaut +1,66 nat sur le quartile trivial puis
environ +3 nats partout ailleurs — **à peu près constant quelle que soit la
difficulté**. Sur une position où la référence est certaine à 99,9 %, le moteur
tombe à 19 %.

Un écart indépendant de la difficulté est la signature d'un **bruit ajouté aux
logits**, non d'une erreur sélective.

## La chaîne causale

Raccordée à la mesure par couche : les couches d'attention **linéaire**
contribuent 11 à 40 % d'erreur quand les couches d'attention pleine restent à
1 %.

    couches linéaires bruitées → activations bruitées → logits bruités
    → même le trivial devient incertain → perplexité ×20
    mais l'argmax reste souvent juste → texte plausible

Le paradoxe qui a résisté toute la journée — un modèle qui génère du texte
cohérent avec une perplexité de 200 — est expliqué sans contradiction.

## Ce que le dossier laisse établi

- le format q3n coûte **8 %** de perplexité pour 1,25 bit de moins par poids ;
- le filet de promotion valait un facteur **4,8** et était le vrai sujet côté
  conversion ;
- **97,5 %** de l'écart à l'étalon est au moteur, dans les couches d'attention
  linéaire, et non dans les formats ;
- l'hypersensibilité des architectures hybrides à la perturbation des poids est
  réelle (**6,1×** le dense) mais vingt-six fois trop petite pour expliquer quoi
  que ce soit ici.

## Ce qui reste non mesuré

Le mécanisme précis dans les couches linéaires. Toutes les causes candidates ont
été éliminées une à une par la mesure — précision réduite, transport, ordre
d'accumulation, type de RoPE, dimension de tête, découpage de la tête de sortie,
frontière de morceau, chemin d'évaluation contre chemin de service. Ce qui
reste est un fait établi sans cause identifiée, et c'est là que le travail
reprend.

---

# Décomposition de l'écart résiduel

Après le correctif du drapeau perdu au transport, il reste 0,3994 nat entre le
NVFP4 et l'étalon. Décomposition par témoins mesurés, chacun dans le cadrage des
584 fenêtres.

| terme | mesure | valeur |
|---|---|---|
| (a) opération de requantification | Q3_K_S → **Q3_K_S**, llama.cpp | **0,0000 nat** |
| (b) changement de grille vers plus fin | Q3_K_S → **Q4_K_S**, llama.cpp | **0,0291 nat** (+3,0 %) |
| (c) biais d'instrument, **dense** | phi-4 int8 contre 6,5988 | 0,0461 nat |
| **(d) résidu** | par soustraction | **0,3243 nat (+38,3 %)** |

**Le témoin (a) vaut exactement zéro** : 9,1831 ± 0,07301 des deux côtés, à la
quatrième décimale. Déquantifier puis requantifier sur la même grille ne coûte
rien — et, au passage, `llama-quantize --allow-requantize` est **idempotent** sur
ce format, ce qui valide toute la chaîne de mesure de la journée sur deux
fichiers produits à des heures différentes.

**Le résidu est réel** : le seuil de significativité est 0,01 nat, on est trente
fois au-dessus. **81 % de l'écart reste inexpliqué.** Et le témoin (b) joue
*contre* cette conclusion plutôt que pour : un coût de format plus élevé
réduirait le résidu d'autant.

Ce que (d) contient, sans hiérarchie établie : tout ce que la chaîne de
conversion fait en plus d'un changement de grille (recherche d'échelles par
canal, rotation de Hadamard, filet de promotions, mélange de formats dans le même
dossier) **et** le biais d'instrument sur architecture **hybride**, jamais
mesuré — (c) vaut pour un dense. Pour absorber 0,324 nat, ce dernier devrait
valoir sept fois le biais dense.

---

# Échelle contre table : ce qui domine sous 3,25 bits

arXiv:2605.24011 affirme que sous quatre bits **le choix de l'échelle domine
celui de la table**. Vérifié sur nos poids, 144 tenseurs d'un tirage disjoint de
tous les précédents, 3,25 bits par poids pour toutes les combinaisons.

| table | échelle | SNR médian | gain | pire | % qui perdent |
|---|---|---|---|---|---|
| spécification | amax (actuelle) | 13,33 | — | — | — |
| spécification | grille optimisée | 15,48 | +2,14 | +1,50 | 0,0 |
| spécification | grille pondérée par la magnitude | 15,16 | +1,81 | +0,88 | 0,0 |
| **Lloyd** | amax | **16,71** | **+3,42** | +3,36 | 0,0 |
| **Lloyd** | **grille optimisée** | **17,64** | **+4,34** | +4,22 | 0,0 |
| Lloyd | grille pondérée | 17,53 | +4,23 | +4,09 | 0,0 |

**L'affirmation est fausse sur nos poids : la table domine l'échelle**, +3,42 dB
contre +2,14. La pondération par la magnitude — le cœur de leur méthode — est
même légèrement *pire* que la recherche simple.

Deux résultats exploitables malgré tout : les deux leviers **s'additionnent**
presque parfaitement (+3,42 puis +0,92, total +4,34), et **aucune combinaison ne
fait perdre un seul tenseur** sur 144.

Réserve de protocole : la recherche d'échelle balaie une grille de facteurs
autour de l'amax et retient le meilleur par bloc ; un ajustement analytique
ferait mieux. Il faudrait qu'il double pour renverser l'ordre.

Leçon générale, et c'est la même que pour la table à niveau zéro : **un résultat
publié vaut pour la distribution et le régime qui l'ont produit.** Ici 2,6 bits
sur un modèle vision-langage-action contre 3,25 bits sur du texte — le mécanisme
ne s'est pas transporté.

---

# Énergie par jeton : une comparaison qu'il ne faut pas faire

Ce document a un temps opposé nos 17,4 J/jeton bruts à une « valeur typique de
1,8 J ». **Cette valeur typique n'existe pas** : la littérature de mesure donne
0,003 à 1 J par jeton selon le couple modèle-carte, soit trois ordres de
grandeur. Trois facteurs dominent et n'étaient pas contrôlés :

- **le lot.** Un même modèle passe de 0,209 J/jeton à lot 128 à 0,151 à lot 512.
  Nos mesures sont à **lot 1**, le pire cas absolu : toute la puissance statique
  se répartit sur un seul jeton. Or le moteur accepte 16 requêtes simultanées par
  défaut — **c'était un choix de protocole, pas une limite** ;
- **les paramètres actifs, non totaux.** Un MoE coûte 3,56 fois moins par jeton
  qu'un dense de même taille totale ;
- **la puissance statique gaspillée** quand l'occupation est faible — exactement
  le profil mesuré ici, cartes inoccupées une seconde sur deux.

On ne peut donc affirmer ni que ce moteur est dix fois trop cher, ni qu'il est
bon. **Un joule par jeton ne veut rien dire sans son lot, ses paramètres actifs
et son taux d'occupation** : c'est la condition à remplir avant toute
publication, et elle vaut aussi pour lire les chiffres des autres.

Sur la méthode elle-même, une inquiétude est levée : le GPU porte 78,7 à 92,5 %
de l'énergie totale du nœud, donc une mesure au compteur NVML ne biaise pas le
résultat d'un facteur qui compte ici.


---

# Le transport d'experts ne change pas un chiffre

Mesuré le 8 septembre 2026 au soir, sur `lfm-8b-a1b-bf16`, protocole figé et
prédictions écrites avant la mesure.

| passe | couches exilées | perplexité | positions | durée |
|---|---|---|---|---|
| A₁ | 0 (placement naturel) | **58,815** | 146 717 | 82 s |
| A₂ | 0, exécution identique | **58,815** | 146 717 | 61 s |
| B | **12**, seul `mlp_storage`/`mlp_exec` change | **58,815** | 146 717 | **257 s** |

**Les trois chiffres sont identiques.** Les deux prédictions sont vérifiées : la
perplexité est reproductible d'une exécution à l'autre, et elle ne bouge pas
quand douze couches passent en mémoire hôte. La troisième issue prévue — un
écart de l'ordre de 5e-3 dû à un changement d'ordre d'accumulation — ne s'est
pas présentée, donc pas de départage en fp32 à faire.

**Et la quatrième colonne dit ce que les trois premières taisent** : la même
passe met **257 s contre 61 et 82**, un facteur trois à quatre. L'exil s'est donc
bel et bien produit — **il se voit dans le temps et pas dans le chiffre**. C'est
la séparation qualité/performance mesurée au lieu d'être postulée : douze couches
sur le PCIe coûtent un facteur quatre en temps et **zéro en qualité**.

**Le contrôle qui rend la mesure valide**, exigé avant le lancement et présent au
journal : `mesure : 12 couches à perceptron exilé (minimum imposé par la
capacité : 0)`. Sans cette ligne, B aurait pu être une troisième exécution de A —
égalité prédite, conclusion « transport innocent », et deux fois la même mesure.
Aucune passe n'a émis `plan réajusté`, et la VRAM libre était identique aux trois
chargements.

## Ce que ça change dans la table des témoins

Les cases **« MoE résident »** et **« MoE exilé »** n'en font plus qu'une : un
modèle partiellement exilé se mesure et s'interprète **sans réserve sur le
transport**. La réserve posée sur le témoin `qwen3-coder-30b-bf16` — 26 couches
sur 48 en mémoire hôte, puis un exil forcé à 30 pour contourner la fragmentation
— est nommée mais **n'affaiblit plus son chiffre**.

## Ce que ça n'établit pas

Que ce soit vrai de **toute** architecture. `lfm-8b-a1b` est hybride à
convolutions courtes : l'invariance porte sur **son** chemin d'experts, pas sur
un GDN ni sur une attention latente. Transporter ce résultat aux autres
architectures serait la faute commise sept fois dans la même journée — un
résultat vrai, appliqué hors de ses conditions.
