# anticitoyen VRAM/RAM (`acvram`)

Une passerelle d'inférence compatible avec l'API OpenAI, qui traite la mémoire
comme une hiérarchie et donne à chaque GPU le format numérique que son silicium
sait le mieux lire.

Conçue pour une machine précise :

| | |
|---|---|
| Processeur | Intel Core i9-14900K (8 cœurs P + 16 cœurs E) |
| Carte mère | ASUS ROG Maximus Z790 Dark Hero |
| Mémoire | 96 Go DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 Go — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 Go — Ampere, `sm_86` |
| Système | Linux Mint 22.3 |

## Les deux idées

**Un format par GPU.** La RTX 5090 possède des tensor cores FP4 ; la RTX 3080 Ti
n'en a pas, et n'a pas non plus de FP8. Aligner les deux sur un format commun
gâcherait la 5090. Le convertisseur écrit donc *deux fois le même modèle*, dans
le format que chaque destination sait réellement exploiter :

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| poids | **NVFP4** — E2M1 + échelle FP8 E4M3 tous les 16 | **INT4** — uint4 + échelle et zéro fp16 tous les 128 |
| bits par poids | 4,50 | 4,16 |
| face au BF16 | ×3,56 plus petit | ×3,85 plus petit |
| mode de calcul | tensor cores FP4 | déquantifié en FP16 dans le noyau, tensor cores FP16 |
| cache KV | INT8 | INT8 |

32 Go de VRAM à 4,5 bits par poids contiennent environ **56 milliards de
paramètres**, contre 16 milliards en BF16. Sur les deux cartes, cela fait
approximativement **78 milliards de paramètres résidents** avant même de
toucher à la mémoire vive.

**La mémoire est une hiérarchie, pas un mur.** Trois étages, et le planificateur
mesure ce que chacun coûte au lieu d'espérer que le modèle tienne :

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Démarrage rapide

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

N'importe quel client OpenAI s'y branche ensuite :

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Bonjour"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Bonjour"}])
```

## Ce que dit `acvram plan`

Le planificateur mérite d'être lancé avant tout téléchargement. Il répond aux
questions qui décident si un modèle est utilisable sur cette machine :

```
$ acvram plan ~/modeles/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  etage   format      capacite      poids         KV  tranche
  cuda:0  nvfp4        30,3 Gio   25,5 Gio   4,5 Gio  couches 0-58
  cuda:1  int4_awq     10,9 Gio    8,7 Gio   1,6 Gio  couches 59-79
  cpu     nvfp4        74,8 Gio    3,4 Gio       0 o  -

  poids au total     37,6 Gio
  lu par jeton       35,1 Gio
  KV par jeton       162,5 Kio  -> 39 843 jetons en cache
  MLP en RAM hote    55-58

  decodage estime    17,8 jetons/s  (lot de 1)
  prefill estime     847 jetons/s
```

Il explore l'espace des configurations au lieu de retenir la première qui
tient, et deux de ses décisions sont assez contre-intuitives pour mériter
d'être énoncées :

* **Il laisse la 3080 Ti inutilisée** quand un modèle tient sur la seule 5090.
  Les tranches d'un pipeline s'exécutent en série : ajouter une étape à
  912 Go/s dans un pipeline à 1790 Go/s ralentit le décodage mono-flux. On force
  avec `--gpus all`.
* **Il rétrécit le cache KV pour garder les poids en VRAM.** Chaque gigaoctet
  donné au cache est un gigaoctet de poids repoussé sur le bus PCIe, et lire un
  poids par le PCIe coûte environ trente fois ce qu'il coûte depuis la VRAM. Sur
  le 70B ci-dessus, ce seul arbitrage fait passer de 2,3 à 17,8 jetons/s.

## Aller vite

Quatre optimisations, chacune vérifiée par une preuve d'équivalence et pas
seulement par un chronomètre : une optimisation qui change la réponse est un
bogue.

### Décodage spéculatif (`--speculative`)

Décoder un jeton avec un lot de taille 1 est limité par la mémoire : la machine
lit tous les poids actifs pour produire un seul jeton. Vérifier K jetons
proposés lit ces mêmes poids **une seule fois**. Deux propositeurs :

* `ngram` (par défaut) — cherche le suffixe courant plus tôt dans le contexte et
  propose ce qui suivait. Ne coûte rien, ne demande aucun modèle. Rentable quand
  la sortie recopie l'entrée : édition de code, RAG, résumé.
* `draft` — un petit modèle sur un second appareil. Sur ce rig, cet appareil est
  la RTX 3080 Ti, que le planificateur laisse volontairement oisive pour tout
  modèle qui tient sur la 5090.

L'acceptation est exacte, pas approchée : une proposition est acceptée avec la
probabilité `min(1, p/q)` et un rejet rééchantillonne dans la partie positive
normalisée de `p - q`. Mesuré sur 40 000 tirages face à un brouillon
volontairement mal calibré, la distribution émise reste à 0,002 de variation
totale de la cible — la spéculation achète de la vitesse, jamais une réponse
différente.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Cache de préfixe (actif par défaut)

Les blocs sont adressés par le hachage *chaîné* de leur tranche de jetons : deux
requêtes qui partagent une consigne système partagent ses blocs, et la seconde
n'a plus à les précalculer. Le chaînage est indispensable : les mêmes seize
jetons dans un contexte différent ne contiennent pas les mêmes clés et valeurs,
et hacher la seule tranche servirait le cache d'une séquence à une autre.

Un bloc libéré dont le contenu reste identifiable rejoint une file LRU plutôt
que la liste des blocs libres : le cache survit ainsi entre les requêtes sans
jamais refuser une allocation qu'il aurait pu servir.

### Calcul de l'étage hôte (`--host-exec`)

Une couche dont les poids résident en RAM peut être copiée vers le GPU ou
calculée sur place. Les deux chemins sont limités par la mémoire et lisent les
mêmes octets : le plus rapide est celui dont le bus est le plus large — le PCIe
5.0 x16 donne environ 54 Go/s, la DDR5 en double canal environ 70 Go/s — et
calculer sur place laisse en outre le GPU libre au lieu de le faire attendre une
copie.

Cela ne vaut que si le processeur lit directement les poids empaquetés sur
4 bits. D'où un petit noyau C++ avec un chemin AVX2 (`acvram_cpu.cpp`, chargé
par ctypes, sans en-têtes Python ni ninja). Même sur sa branche **scalaire** de
repli, il bat `dequantize() @ x` d'un facteur 1,44 en INT4 et 3,21 en NVFP4,
parce que ce dernier écrit d'abord une copie 32 bits de toute la matrice.

Sur Mistral-Large-123B, l'estimation du planificateur passe de 1,35 à
2,42 jetons/s.

### Précision mixte (`--snr-floor`, éteinte par défaut)

Le convertisseur mesure le rapport signal/bruit en sortie de couche pour chaque
tenseur et peut promouvoir vers un format plus large ceux qui tombent sous
`--snr-floor`, dans la limite de 15 % des tenseurs et d'un prix plafond
(`--promotion-cout-max`, en mébioctets ajoutés).

Le plancher vaut **zéro par défaut** : rien n'est promu. Le décodage est limité
par la bande passante mémoire, et la mesure sur `Huihui-Qwen3.8-27B` tranche —
un plancher de 25 dB coûte 13,4 % de mémoire et 10,6 % de débit (18,50 Gio et
41,8 t/s contre 16,02 et 46,2) pour 2,0 % de perplexité (42,591 contre 43,447,
corpus de 16 383 jetons). `--snr-floor 25` rétablit l'ancien comportement quand
la qualité prime sur la vitesse.

### Et `acvram eval`

Le rapport signal/bruit et le cosinus des logits sont des approximations.
`acvram eval REP [REP ...]` mesure la perplexité par fenêtre glissante, pour
qu'un choix de format se tranche sur des preuves :

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## Points d'entrée HTTP

| point d'entrée | notes |
|---|---|
| `POST /v1/chat/completions` | flux SSE ou réponse unique ; utilise le gabarit de conversation du modèle |
| `POST /v1/completions` | invite en texte ou en identifiants de jetons |
| `POST /v1/embeddings` | états cachés finaux moyennés, normalisés L2, `dimensions` respecté |
| `GET /v1/models` | plus un bloc `acvram` : formats, appareils, capacité du cache KV |
| `GET /health`, `GET /metrics` | débit de décodage, occupation des blocs KV |

Les noms de champs de ces réponses restent en anglais : c'est le protocole
OpenAI, et les traduire romprait tous les clients existants.

## D'où viennent les chiffres

Chaque valeur citée ci-dessus est produite par du code de ce dépôt et vérifiée
par `pytest`. Mesures faites sur processeur avec les noyaux de référence :

| format | bits/poids | SNR des poids | cosinus des logits vs BF16 |
|---|---|---|---|
| BF16 | 16,00 | — | 1,0000 |
| INT8 | 8,19 | 44,6 dB | 0,9998 |
| NVFP4 | 4,50 | 20,4 dB | 0,9664 |
| INT4 | 4,16 | 20,0 dB | 0,9427 |
| INT4 + Hadamard | 4,16 | 21,0 dB | 0,9582 |

Deux constats issus de ces mesures ont changé les valeurs par défaut :

* **Une rotation de Hadamard aide l'INT4 et pas le NVFP4.** Les groupes de 128
  de l'INT4 ne peuvent pas absorber un canal aberrant isolé, si bien qu'étaler
  les valeurs extrêmes vaut une transformée en n log n par activation. Les blocs
  de 16 du NVFP4 portent déjà leur propre échelle. D'où `--hadamard auto`, qui
  ne l'applique qu'à l'INT4.
* **L'INT8 bat le FP8 E4M3 pour le cache KV**, 44 dB contre 32 dB à taille
  identique, parce qu'une échelle par (jeton, tête) fournit déjà la plage
  dynamique pour laquelle le FP8 dépense des bits d'exposant. Les deux cartes
  utilisent donc un cache KV en INT8, même si la 5090 saurait faire du FP8.

## Documentation

* [`REPRISE.md`](REPRISE.md) — **reprendre le projet sur une autre machine**
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — comment les pièces s'assemblent
* [`docs/MATERIEL.md`](docs/MATERIEL.md) — régler cette machine précise
* [`docs/FEUILLE-DE-ROUTE.md`](docs/FEUILLE-DE-ROUTE.md) — **ce qui n'est pas fait**, à lire en premier
* [`CONVENTIONS.md`](CONVENTIONS.md) — conventions de travail sur le code (langue, style, contrôles avant de pousser)

## État

Version 0.2.0. Écrit avant que la machine cible ne soit disponible : tous les
chemins de code sont exercés sur processeur, aucun n'a encore tourné sur une
5090. Les noyaux processeur *sont* compilés et testés ; les noyaux CUDA n'ont
jamais vu nvcc. Voir [`docs/FEUILLE-DE-ROUTE.md`](docs/FEUILLE-DE-ROUTE.md) pour
ce que cela implique exactement.

67 tests, sur processeur, déterministes, environ une minute : `pytest -q`.

## Licence

GPL-3.0 ou ultérieure.
