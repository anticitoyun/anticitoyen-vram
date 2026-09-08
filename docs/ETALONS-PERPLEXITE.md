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
