# La perplexité : protocole, et ce qui l'entoure

Écrit le 8 septembre 2026, après deux jours pendant lesquels trois sessions ont
interprété un chiffre qui ne mesurait pas ce qu'elles croyaient.

## Le cadrage fait le chiffre

Une perplexité sans son cadrage ne veut rien dire. Deux nombres obtenus sur le
même modèle et le même corpus, avec deux cadrages différents :

| Cadrage | Perplexité | Positions notées |
|---|---|---|
| fenêtre 512, contexte minimal 256 | **7,233** | 16 320 |
| fenêtre 512, contexte minimal 0 | 9,525 | 32 704 |
| corpus interne, tel qu'il servait | 15,241 | 249 |

Même modèle (nemo-12b-thinking-exl3, 7,55 bits par poids), même moteur. **Un
facteur deux, uniquement par le cadrage.**

La cause se lit dans la décomposition par tranche de contexte, sur
`wiki.test.raw` :

| Contexte disponible | Perplexité | Positions |
|---|---|---|
| à partir de 0 jetons | **3597,8** | 512 |
| à partir de 8 jetons | 35,5 | 1 536 |
| à partir de 32 jetons | 10,5 | 6 144 |
| à partir de 128 jetons | 7,6 | 24 512 |

Cinq cent douze positions sur trente-deux mille — un virgule six pour cent —
tirent la moyenne de 7,6 à 9,5. Un jeton prédit sans contexte coûte une dizaine
de nats quel que soit le modèle : ce n'est pas une propriété du modèle, c'est
une propriété de la position. Sur un corpus court, où ces positions pèsent une
fraction bien plus grande, l'effet explose.

## Le protocole

`llama.cpp` note à partir de `n_ctx/2` (`tools/perplexity/perplexity.cpp:541`,
`const int first = n_ctx/2;`) : 256 positions par fenêtre de 512, chacune ayant
au moins 256 jetons devant elle. Pour qu'un chiffre acvram soit comparable :

```
acvram eval <modele> --corpus wiki.test.raw \
    --window 512 --stride 512 --min-context 256
```

Étalon de référence : `wiki.test.raw`, sha256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`,
1 290 590 octets, conservé dans `/mnt/AI_GENERATOR/corpus/`.

Trois règles qui vont avec :

1. **Comparer au cumul à la même fenêtre, jamais au chiffre final.** Le
   wikitext oscille avant de se stabiliser : le cumul llama.cpp vaut 8,33 à la
   16ᵉ fenêtre, 6,82 à la 32ᵉ, 8,52 à la 64ᵉ, 9,17 à la 128ᵉ et 9,18 à la 584ᵉ.
   Comparer une mesure sur 64 fenêtres au 9,18 final impute 7 % au format alors
   qu'il n'y a que du corpus.

2. **Un chiffre tiré de trop peu de positions n'est pas un chiffre.** Le corpus
   interne fait moins de 300 jetons ; recadré à 256, il n'en laisse presque
   aucune. L'avertissement est attaché au résultat, JSON compris, et porte le
   compte en clair — un chiffre se recopie sans son écran.

3. **Une comparaison inter-instruments doit être étalonnée avant de servir.**
   Un chiffre acvram comparé à un étalon llama.cpp porte l'écart entre les deux
   implémentations, et cet écart s'imputerait entièrement au format mesuré.
   L'étalonnage se fait sur un modèle que les deux outils lisent, même corpus,
   même cadrage : l'écart obtenu *est* le biais, en clair. Réserve à garder :
   la conversion acvram change les poids, donc cet écart mêle biais d'instrument
   et coût de conversion — c'est une borne supérieure, et une borne supérieure
   suffit à décider si le sujet existe.

## Les bornes sont identiques a celles de llama.cpp — verifie sur la source

Question posee et refermee le 8 septembre : le harnais note-t-il les memes
positions que la reference ? **Oui, position par position.** La verification
porte sur `tools/perplexity/perplexity.cpp` de llama.cpp e34f042.

| | llama.cpp | acvram |
|---|---|---|
| premiere position | `first = n_ctx/2` = 256 | `first_new = min_context` = 256 |
| nombre notes | `n_ctx - 1 - first` = **255** | `logits[:-1][256:]` = **255** |
| logits utilises | 256 a 510 | 256 a 510 |
| jetons predits | **257 a 511** | **257 a 511** |
| fenetre partielle | jamais (`tokens.size() / n_ctx`, division entiere) | ecartee par `first_new >= logits.shape[0]` |
| agregation | `nll /= count` puis `exp` | `total_nll / counted` puis `exp` |

Le piege : on lit `first = n_ctx/2` et on en deduit 256 positions notees a
partir du jeton 256. C'est faux deux fois — `n_ctx - 1 - first` en donne 255,
et la cible est `tokens[i+1]`, decalee d'un. Les deux erreurs se compensent
exactement avec le `logits[:-1]` et le `first_new` du harnais.

Verification independante du compte : phi-4 rend 144 075 positions, soit
565 x 255, et l'etalon llama.cpp annonce 565 fenetres.

**Consequence : un ecart entre les deux outils ne vient pas de la
comptabilite.** Il vient du forward, ou du cout reel de la requantification —
et une seule mesure les separe : le meme modele converti en bf16 pur, sans
aucune quantification, evalue au meme cadrage.

## Un chiffre juste peut porter une description fausse

Le 8 septembre, trois fois dans la même journée, un chiffre exact a été
accompagné d'une description qui aurait fait conclure de travers :

- une taille de fichier annoncée au lieu d'être mesurée, alors que l'empreinte
  sha256 à côté, elle, était mesurée ;
- une cible de comparaison prise au terme d'une série alors que la mesure
  portait sur son début ;
- une tranche de contexte étiquetée « à partir de 128 jetons » alors que le
  filtre en imposait 256 — donc annonçant un objet plus facile que celui qui
  avait été mesuré ;
- un modèle demandé en int8 et sorti avec ses 72 projections de perceptron en
  q3n, dans un dossier nommé « témoin-int8 », la bascule ayant été annoncée
  dans un journal détaché que personne n'a lu ;
- un écart de bornes de notation calculé exactement — et sur une prémisse
  inventée, alors que la source de référence était sur le disque à la version
  exacte. Cinq minutes d'arithmétique juste sur une lecture qui n'avait pas
  eu lieu.

Les trois chiffres étaient bons. C'est ce qui les entourait qui était faux, et
aucun des trois n'aurait été rattrapé par une vérification du calcul.

**La règle qui en sort :** vérifier une valeur ne suffit pas, il faut vérifier
sa description — son unité, ses conditions, son domaine de validité, et si elle
a été mesurée ou déduite. Dans le doute, un champ mesuré et un champ décrit ne
se rangent pas au même endroit.

C'est pourquoi le résultat porte désormais son cadrage : fenêtre et contexte
minimal sont imprimés en tête du tableau et présents dans le JSON. Sans eux,
7,23 et 137 se lisent comme le même objet.

## Ce qui n'est pas mesuré

- La perplexité du NVFP4 sain dans le cadrage corrigé. Deux tentatives mortes
  par manque de mémoire vive : 44 Go sur une machine partagée, le cache de
  pages n'étant pas rendu après une lecture massive antérieure. **Sans ce
  chiffre, l'écart entre formats reste inconnu dans le bon cadrage.**
- Le biais entre le harnais acvram et llama.cpp.
- Le cadrage explique un facteur deux. Le 137 rapporté sur un modèle sain reste
  donc inexpliqué aux neuf dixièmes : un témoin d'un autre modèle borne
  l'instrument, il ne referme pas le dossier.
