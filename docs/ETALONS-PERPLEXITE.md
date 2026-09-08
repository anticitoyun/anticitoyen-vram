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
