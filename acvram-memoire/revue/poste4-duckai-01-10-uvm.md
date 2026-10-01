# duck.ai 01/10 — UVM/VMM/HMM pour un outil de dépassement VRAM (ordre chef)

Sources primaires (docs NVIDIA, arXiv, GitHub, lues directement ou vérifiées via web search)
puis 3 modèles duck.ai raisonnement (Gemma 4 31B, GPT-5.6 Luna, gpt-oss 120B, web search activé).
Convergence forte des trois modèles entre eux et avec les sources primaires — aucune
contradiction relevée, pas de conclusion prise à la place du groupe.

## Q1 — sm_120 grand public sous Linux, limites

**Pris en charge, avec conditions précises** : HMM (mémoire unifiée complète, allocations
système ordinaires `malloc`/`mmap` accessibles au GPU, oversubscription autorisée) nécessite
noyau Linux ≥ 6.1.24/6.2.11/6.3, pilote CUDA ≥ 535, et **le pilote `nvidia-open` (Open Kernel
Modules)** — Blackwell (sm_120) n'est supporté QUE par les modules ouverts, plus par le pilote
propriétaire. Vérification : `nvidia-smi -q | grep Addressing` → `Addressing Mode : HMM`. RTX
5090 (sm_120) et 3080 Ti (sm_86) satisfont toutes deux le seuil architectural documenté
(compute capability ≥ 7.5). Source : CUDA Programming Guide (sections Unified Memory / HMM,
nvidia.com) ; confirmation indépendante que Blackwell exige les modules ouverts (Phoronix,
documentation NVIDIA CUDA Features Archive 12.8).

**Taille de page** : pas une garantie universelle de 2 Mio — granularité variable ~4 Kio à 2 Mio
selon chemin d'allocation, alignement, noyau, pilote. Les pages de 2 Mio sont courantes dans le
chemin UVM CUDA. Le papier vAttention (arXiv 2405.04437) note explicitement que
`cudaMallocManaged` utilise par défaut des pages de 2 Mio, cause de fragmentation sévère, et que
l'UVM standard ne fournit ni libération partielle ni aliasing mémoire — c'est la raison nommée
du rejet de l'UVM brute par vLLM (voir Q3).

**Défauts de page** : coût = temps de transfert PCIe + interception du défaut + synchronisation
pilote + mise à jour des tables de pages, pas seulement la bande passante brute.
`cudaMemPrefetchAsync` existe précisément pour sortir ce coût du chemin critique en démarrant la
migration avant le kernel — mais reste une indication au pilote, pas un contrat de résidence.

**Graphes CUDA** : non fondamentalement incompatibles, mais la résidence/migration devient un
effet dynamique non représenté comme dépendance explicite dans le graphe. Bonnes pratiques
documentées : allouer la mémoire managée AVANT la capture et la réutiliser à l'intérieur ;
`cudaMalloc`/`cudaFree` interdits pendant la capture (Global/ThreadLocal) ; un défaut de page
peut survenir pendant le replay avec une latence peu déterministe. Pour un pipeline de couches :
capturer le calcul avec des pointeurs déjà stables en VRAM, gérer transferts/remappages hors
graphe ou en segments séparés.

## Q2 — débit de migration, poids séquentiels couche par couche

**Rien de publié et vérifié spécifiquement pour RTX 5090/PCIe 5.0/lecture séquentielle de poids
LLM** — ni dans nos recherches ni dans celles des 3 modèles. À ne pas combler par un chiffre
théorique PCIe présenté comme mesuré.

Ce qui est publié, hors de notre carte (tendances, pas des chiffres transposables tels quels) :
- sans prefetch, démand-paging UVM tombe en régime limité par PCIe (~8,2 Go/s contre 242 Go/s
  théorique sur une plateforme mesurée, soit ~30×) ; avec `cudaMemPrefetchAsync`, le
  ralentissement géométrique moyen passe de 95,8 % à 0,7 % — quasi la vitesse d'une copie
  épinglée ;
- par défaut, un défaut de page transfère en moyenne ~71 Kio, ~189 ms pour 2 Gio (30 078
  transferts) ≈ 10 Go/s ;
- à 50 % de sursouscription, les `cudaMemAdvise` donnent jusqu'à 25 % de mieux que l'UVM nu ;
  le prefetch donne jusqu'à 50 % de mieux sur certaines plateformes PCIe, peu d'effet sur
  NVLink — synthèse relayée du blog NVIDIA « Improving GPU Memory Oversubscription
  Performance », page non lue intégralement nous-mêmes (fetch bloqué sur la barre latérale),
  donc à confirmer en lisant la page directement si le chiffre doit servir de base de décision ;
- granularité de chunk pour s'approcher du débit DMA explicite : ~256 Kio pour ~90 % du pic
  PCIe 3.0 x16 dans une étude publique ; à 64 Kio, l'efficacité tombe sous 50 % dans la même
  étude.

Le seul chiffre publié proche du domaine LLM est une mesure de **débit d'allocation/mapping
VMM**, pas de migration UVM ni de transfert host-to-device : vAttention rapporte jusqu'à
7,6 Go/s de mémoire physique allouée par GPU (pages de 64 Kio), contre ~750 Mo/s requis par la
charge de décodage — ne pas confondre avec un débit PCIe.

## Q3 — les moteurs d'inférence utilisent-ils UVM ou VMM pour le déchargement ?

| Moteur | UVM standard pour les poids | VMM pour le KV/cache | Mécanisme réel observé |
|---|---|---|---|
| vLLM | Non — demande explicite fermée **« Closed as not planned »** (issue GitHub vllm-project/vllm **#10267**, « [Feature]: Support for NVIDIA Unified memory ») | PagedAttention (gestion logique/physique maison, pas l'API VMM `cuMemMap`) | poids/activations en allocations GPU classiques, KV cache paginé, offload CPU explicite selon fonctionnalité |
| llama.cpp | Existe mais ciblé GPU intégrés, pas pour l'oversubscription performante de carte discrète : `cudaMallocManaged`/`hipMallocManaged` via la variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY` (remplace l'ancien flag de compilation `GGML_HIP_UMA`, unifié par la PR **#12934** « CUDA/HIP: Share the same unified memory allocation logic ») — note AMD : mémoire « fine-grained » par défaut lente, « coarse-grained » plus rapide | Non, mécanisme central = `mmap` des GGUF + placement explicite de couches | couches placées explicitement, buffers et copies maison |
| SGLang | Aucune preuve vérifiée d'UVM standard (3 modèles + nos recherches, aucun n'en trouve) | Pas de pool VMM généralisé type vAttention trouvé | RadixAttention, cache de préfixe + offload CPU/stockage par paliers explicites |
| DeepSpeed (ZeRO-Infinity/ZeRO-Inference) | Aucune preuve | Non, moteur d'offload dédié | offload CPU/NVMe explicite, tiling des opérateurs |
| FlexGen | Non | Pas de `cuMemMap` pour un pool VRAM — fichiers tenseurs mappés côté stockage seulement | ordonnancement « zig-zag » GPU/CPU/disque explicite, recouvrement E/S-calcul par streams |
| PowerInfer | Non | Non | partition statique neurones chauds (GPU, préchargés) / froids (CPU), pas de migration de pages |

**Gain/perte contre un déchargement explicite** : aucun A/B publié trouvé pour l'un de ces six
moteurs comparant explicitement sa stratégie à une bascule vers l'UVM. L'argument pour
l'explicite reste structurel, pas chiffré : l'unité de migration explicite (tenseur/MLP/expert/
couche) est sémantiquement pertinente, contre une migration par page qui ignore les frontières
de calcul ; le pipeline explicite peut précharger pendant le calcul courant et éviter de
rapatrier des poids qui ne seront plus utilisés.

## Q4 — VMM (`cuMemMap`) pour un pool VRAM extensible/défragmentable

**vAttention** (ASPLOS'25, Microsoft Research India, arXiv **2405.04437**) est le cas le mieux
documenté pour un KV cache LLM : réserve une plage virtuelle contiguë potentiellement énorme
(exemple cité : 12 To virtuels par worker pour Yi-34B) via `cuMemAddressReserve`, puis mappe
physiquement à la demande (`cuMemCreate`/`cuMemMap`) — garde la contiguïté virtuelle pour
réutiliser des noyaux d'attention existants (dont FlashAttention-3 hors boîte) sans réécriture
pour PagedAttention. **Gain publié : deux chiffres trouvés dans des lectures différentes du
même papier, à vérifier directement avant de choisir lequel citer** — synthèse moteur de
recherche de l'abstract ASPLOS : « jusqu'à 1,36× de débit en prefill, jusqu'à 42 % de latence
en moins » ; lecture directe du corps de l'article par Luna : « jusqu'à 1,23× de débit contre
des noyaux fondés sur PagedAttention (FlashAttention/FlashInfer) ». Les deux proviennent
d'arXiv 2405.04437 mais ne sont probablement pas la même comparaison (benchmarks différents
dans le papier) — ne pas trancher sans relire le papier soi-même. vAttention n'est pas un
simple usage nu des API VMM : il étend l'allocateur PyTorch avec des optimisations LLM
spécifiques (chevauchement allocation/calcul, pré-allocation spéculative, réclamation
différée) pour absorber la latence des appels VMM (ronde-trip noyau par allocation).

**PyTorch `expandable_segments:True`** (`PYTORCH_CUDA_ALLOC_CONF`) : s'appuie aussi sur `cuMemMap`
— réserve une grande plage virtuelle, y mappe des segments physiques à la demande, réduit la
fragmentation et les OOM limites. **Aucun chiffre de gain publié et vérifié spécifiquement pour
le débit d'un cache KV ou d'une inférence LLM** n'a été trouvé pour ce mécanisme — à ne pas
inventer. Coût documenté : surcoût d'exécution du mapping mémoire au niveau page.

vLLM et SGLang : aucune preuve vérifiée d'un remplacement général de leur PagedAttention/
allocateurs internes par un pool VMM type vAttention à ce jour dans les sources consultées.

## Désaccord signalé

Aucun désaccord factuel entre les 3 modèles — convergence sur les six moteurs (aucun n'utilise
l'UVM standard comme mécanisme principal), sur le rejet documenté de l'UVM par vLLM, et sur
vAttention comme référence VMM pour le KV cache. Seule divergence : le chiffre de gain
vAttention (1,36×/42 % contre 1,23×), signalée ci-dessus plutôt que tranchée.

**RESTE** : rien en cours après ce lot. Prochaine reprise : ordre de chef ou repos.
