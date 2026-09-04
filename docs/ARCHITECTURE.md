# Architecture

## Le chemin que suit un modèle

```
  point de contrôle Hugging Face (safetensors, bf16)
        |
        |  acvram plan     ->  où va chaque tenseur ?
        v
  plan de placement  ------------------------------+
        |                                          |
        |  acvram convert  ->  quantifie selon     |
        v                      l'appareil visé     |
  fragments acvram + manifest.json                 |
        |                                          |
        |  acvram serve                            |
        v                                          v
  chargeur ----> modèle placé ----> moteur ----> HTTP OpenAI
```

Le plan est calculé **avant** la conversion et rangé **dans** le manifeste du
modèle converti. C'est ce qui supprime toute une classe de bogues : au
chargement, la question « ce GPU sait-il lire ce tenseur ? » ne se pose jamais,
parce que le tenseur a été écrit pour ce GPU.

## Le planificateur

`memory/tiering.py`. À partir d'un `ModelSpec` et d'un `Rig`, il produit un
`Plan`.

Il raisonne sur un modèle de décodage limité par la mémoire : produire un jeton
revient à lire une fois chaque poids *actif*, donc le temps vaut

```
  somme sur les couches de :
      résident  ->  octets_actifs / bande_passante_vram
      streamé   ->  max(octets_actifs / bande_passante_pcie,
                        octets_actifs / bande_passante_vram)   # le préchargement recouvre
  + un saut par le tampon hôte à chaque changement de GPU
```

`octets_actifs` vaut toute la couche pour un bloc dense, et seulement les
experts routés pour un bloc à mélange d'experts — ce qui explique à lui seul
pourquoi un modèle MoE de 235 milliards de paramètres et un modèle dense de 70
se comportent si différemment sur la même machine.

Quatre décisions, dans cet ordre :

1. **Budget du cache KV.** Pris en premier, parce qu'il croît avec le trafic
   alors que les poids, non.
2. **Tranches du pipeline.** Des plages de couches contiguës par GPU,
   dimensionnées pour que le pipeline ne franchisse qu'une seule fois une
   frontière entre cartes. Les cartes GeForce n'ont pas de NVLink et NVIDIA y
   désactive le pair-à-pair PCIe : un franchissement transite donc par la
   mémoire hôte épinglée — négligeable pour un état caché par jeton, ruineux si
   cela arrivait à chaque couche.
3. **Placement de l'attention.** Épinglée sur le GPU de sa tranche dès qu'elle y
   tient. Elle est petite et se trouve sur le chemin critique de la latence.
4. **MLP et experts.** Remplissent la VRAM de l'avant vers l'arrière ; le reste
   part en RAM hôte. La VRAM qui subsiste devient un cache LRU d'experts
   fréquents.

`auto_plan` explore ensuite le produit (nombre de GPU utilisés) × (fraction de
VRAM pour le cache KV) et classe par débit de décodage estimé, en rejetant toute
configuration qui déborde ou qui ne tient pas un contexte complet.

## Formats

| | NVFP4 | INT4 |
|---|---|---|
| élément | E2M1 (4 bits) | uint4 |
| échelle | FP8 E4M3, une pour 16 | FP16, une pour 128 |
| point zéro | aucun (symétrique) | uint4, un pour 128 |
| échelle globale | FP32 par tenseur | aucune |
| bits par poids | 4,50 | 4,156 |
| déquantification | `niveau × e4m3(bloc) × globale` | `(q − zéro) × échelle` |

L'échelle globale en FP32 du NVFP4 existe parce que l'E4M3 sature à 448 : on la
choisit égale à `amax / (6 × 448)`, de sorte que la plus grande échelle de bloc
atterrisse exactement sur ce plafond, quelle que soit la dynamique du tenseur.

## Noyaux de calcul

Deux points d'entrée par format :

* `*_dequant` — matérialise la matrice dans un type de calcul, puis laisse
  cuBLAS faire le produit. C'est le chemin du **prefill** : le coût de
  déquantification s'amortit sur tout le lot, et cuBLAS bat tout produit écrit à
  la main.
* `*_gemv` — déquantification et multiplication fusionnées, pour un lot de 1 à
  8. C'est le chemin du **décodage**, purement limité par la mémoire : il s'agit
  de lire les poids 4 bits depuis la mémoire globale sans jamais en écrire une
  copie 16 bits.

Le seuil entre les deux est `gemv_threshold=8` dans `kernels/__init__.py`.

`kernels/__init__.py` compile l'extension au premier usage, en n'émettant du
code que pour les architectures présentes. Si la compilation échoue pour une
raison quelconque, il avertit une fois et retombe sur l'implémentation PyTorch
de référence — lente, mais numériquement identique, et suffisante pour que la
suite de tests s'exécute n'importe où.

## Moteur

`engine/runner.py` fait tourner un lot continu : à chaque étape il admet autant
de requêtes en attente que les blocs KV libres le permettent, précalcule chacune
séparément (une longue invite mêlée à un lot de décodage bloquerait tout ce qui
la suit), décode un jeton pour tout ce qui tourne, et libère les blocs d'une
séquence dès qu'elle s'arrête.

Il est monothread à dessein. Les couches du modèle sont réparties sur deux GPU
et la mémoire hôte, et une étape les touche en série ; des fils d'exécution se
disputeraient les mêmes appareils sans ajouter de parallélisme. La concurrence
vient du lot, pas des fils. Le serveur asynchrone fait le lien avec un fil
d'arrière-plan et une file par requête.

## Poids streamés

`engine/layers.py:StreamedWeight`. Les poids résidant en RAM vivent en mémoire
**épinglée** — une mémoire paginable forcerait le pilote à sérialiser la copie
et le recouvrement disparaîtrait — et sont copiés sur un flux CUDA annexe vers
un double tampon. `ACVRamModel.forward` lance le transfert de la couche *i+1*
avant d'exécuter la couche *i*, de sorte qu'une couche streamée coûte
`max(copie, calcul)` et non leur somme. C'est exactement ce que suppose le
modèle de coût du planificateur : si l'un change, l'autre doit changer.

## Cache de préfixe

`memory/kvcache.py:BlockAllocator` est à la fois la liste des blocs libres et le
cache de préfixe, parce que les deux se disputent la même ressource.

Le contenu d'un bloc est déterminé par les jetons qui l'ont produit *et* par
tous ceux qui précèdent, d'où un hachage chaîné :

```
h_0 = hachage((0,     jetons[0:16]))
h_1 = hachage((h_0,   jetons[16:32]))
h_i = hachage((h_{i-1}, jetons[16i:16i+16]))
```

Le chaînage n'est pas décoratif. Les mêmes seize jetons apparaissant dans deux
contextes différents ne produisent pas les mêmes clés et valeurs, puisque
l'attention a vu une histoire différente ; hacher la seule tranche servirait
volontiers le cache d'une séquence à une autre.

Seuls les blocs **complets** sont publiés. Un bloc à moitié rempli, retrouvé par
un hachage qui décrit un contenu qu'il ne porte pas encore, livrerait à une
requête ultérieure des clés et des valeurs jamais écrites.

À la libération, un bloc dont le contenu reste identifiable part en fin de file
LRU au lieu de rejoindre les blocs libres, et n'est recyclé que lorsque la
réserve s'épuise — le cache survit donc entre les requêtes sans jamais refuser
une allocation qu'il aurait pu servir.

Un bloc est toujours retenu lors d'une correspondance : une requête dont
l'invite est entièrement en cache a tout de même besoin d'un jeton à faire
traverser le modèle, sans quoi il n'y a rien pour produire des logits.

La publication a lieu **avant** que les blocs ne soient rendus. L'ordre inverse
— celui d'origine — faisait que `_finish` vidait `seq.blocks` quelques lignes
avant que `_register_complete_blocks` ne cherche à publier : le cache ne
recevait jamais rien, et la réserve de cache KV en mémoire hôte, alimentée par
l'éviction des blocs identifiables, restait inutilisée.

### Hybrides à récurrence linéaire

Les blocs KV ne suffisent pas à reprendre une invite sur un modèle dont
certaines couches portent un état récurrent : cet état vit hors du cache
paginé, dans `Engine.gdn_states`. Le cache y était donc simplement coupé.

Il repose désormais sur des **instantanés** de cet état, pris aux frontières
régulières du prefill et rangés en mémoire hôte épinglée. Sur un 27B, un
instantané pèse 150 Mio pour quarante-huit couches ; trois sont conservés, en
éviction par ancienneté.

Le prefill se coupe une fois, à la plus grande frontière multiple du pas
d'instantané strictement intérieure à l'invite. Ce choix ne dépend que de la
longueur de l'invite, donc deux requêtes partageant une amorce tombent sur la
même frontière tant qu'elles restent dans la même tranche.

À l'admission, l'appariement des blocs est **plafonné** à la frontière dont on
tient l'instantané, et un appariement partiel est rejeté en entier : l'état
récurrent et les clés doivent décrire exactement la même position, faute de quoi
la reprise produirait des logits fondés sur deux histoires différentes.

Le découpage du prefill change l'ordre des calculs récurrents ; la sortie
diverge donc légèrement d'un prefill monolithique, au même titre qu'un
changement de taille de lot.

## Décodage spéculatif

`engine/speculative.py`. Le lot de vérification est `[dernier jeton produit] +
[K propositions]`, présenté aux positions absolues `n-1 .. n+K-1`. Fournir le
dernier jeton produit n'est pas un surcoût : ses clés et valeurs n'ont jamais
été écrites, puisqu'un jeton n'entre dans le cache qu'au moment où on le
présente. Les K+1 positions donnent donc exactement les K+1 prédictions
nécessaires.

Les positions rejetées laissent des entrées périmées dans le cache au-delà de la
longueur de la séquence. Rien ne lit au-delà de `seq_len`, et l'étape suivante
les écrase.

L'acceptation suit la règle de rejet standard. Pour un propositeur sans
distribution (les n-grammes), q est une masse de Dirac sur la proposition : la
probabilité d'acceptation vaut donc `p(x)`, et un rejet rééchantillonne dans `p`
privé de ce jeton — ce qui est exact, et non une approximation.

### Trois brouillons

- **n-grammes** (défaut) : ne coûte rien et paie quand la sortie recopie
  l'entrée. Adaptatif — il se met en pause quand le gain retombe.
- **modèle brouillon** : un modèle entier, avec son propre cache paginé.
  Mesuré ici, il fait tomber le débit de 152 à 30 t/s malgré 78 % d'acceptation
  sur du code : le brouillon coûte trop cher pour ce qu'il rapporte.
- **tête MTP** (`engine/mtp.py`) : la couche `nextn` que portent Qwen3.5 et
  suivants, DeepSeek compris, et que la conversion jetait. Elle mélange le
  plongement du jeton émis et l'état caché de la position précédente, puis
  applique un bloc de transformeur :

  ```
  h' = eh_proj( [ enorm(plongement(t)) ; hnorm(h) ] )
  logits = lm_head( shared_head_norm( bloc(h') ) )
  ```

  L'ordre de la concaténation n'est pas indifférent : plongement d'abord donne
  50 % de prédictions justes en forçage enseignant, l'ordre inverse en donne
  zéro. Cette tête est un soixante-quatrième d'un 27B — mais elle n'est pas
  rentable en l'état : son cache se pollue de ses propres états au fil du
  brouillonnage, et chaque jeton proposé traverse `lm_head` en entier, hors
  graphe CUDA. Disponible par `--speculative mtp`, non activée par défaut.

## L'étage hôte comme appareil de calcul

`DecoderLayer` porte deux appareils : l'attention s'exécute sur `self.device`,
le MLP sur `self.mlp_device`. Quand le planificateur juge qu'un MLP résidant en
RAM vaut mieux calculé sur place, seul l'état caché traverse le bus —
`[jetons, dimension]`, quelques kilooctets par jeton décodé face à des
gigaoctets de poids.

La décision se prend dans `plan_placement`, en comparant le lien hôte mesuré du
GPU exécutant à `PlannerOptions.host_compute_gb_s`. `acvram bench --what
bandwidth` affiche les deux nombres et la recommandation qui en découle.

## Répartition des noyaux

Trois implémentations, numériquement identiques :

| implémentation | fichier | quand |
|---|---|---|
| CUDA | `acvram_kernels.cu` | poids sur un GPU |
| processeur | `acvram_cpu.cpp` (ctypes) | poids en RAM hôte |
| référence | `quant/*.py` | tout ce qui n'a pas pu se compiler |

Le GEMV CUDA tire sa vitesse de trois choses : des chargements `uint2` portant
16 poids empaquetés (exactement un bloc d'échelle NVFP4, et un nombre entier de
groupes INT4, si bien qu'un fil ne chevauche jamais une frontière d'échelle) ;
quatre lignes de sortie par bloc, de sorte que la tranche d'activation est lue
une fois et réutilisée ; et un découpage de la réduction sur K quand la matrice
est trop courte pour occuper l'appareil, ce qui est toujours le cas des petites
projections d'un bloc d'attention à requêtes groupées.
