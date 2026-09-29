# Q37 — duck.ai, 27/09 (non-déterminisme processus-à-processus à b=1, pour poste6 276i et chef)

Question : moteur PyTorch/CUDA maison, RTX 5090 sm_120, graphes CUDA capturés, kernels Triton et CUDA
maison, cuBLAS, flash-attn. À b=1 fixe (pas de variation de lot), mêmes invites, même code, deux
processus serveur neufs successifs divergent sur 1 invite/5 par bascule de jeton. **Ce n'est pas le
phénomène batch-invariance** (Thinking Machines / SGLang, cf. Q35-adjacent) — c'est un non-déterminisme
entre deux lancements du même binaire, pas entre deux tailles de lot au sein d'un même processus.

Sources primaires lues avant duck.ai : blog Thinking Machines "Defeating Nondeterminism in LLM Inference"
(sept. 2025), blog LMSYS/SGLang "Towards Deterministic Inference in SGLang" (sept. 2025), docs PyTorch
Reproducibility (`CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms`,
`torch.backends.cudnn.benchmark/deterministic`), issue GitHub `triton-lang/triton#9368`, issue GitHub
`vllm-project/vllm#58899`. 3 modèles duck.ai (GPT-5.6 Luna, gpt-oss 120B, Gemma 4 31B).

## Verdict par cause

### (a) Autotuning Triton — cause la plus probable, établie

Un kernel `@triton.autotune` chronomètre plusieurs configurations **au premier appel** et retient le
gagnant. Si deux configs sont proches en performance, le gagnant peut dépendre de l'état exact de la
machine à cet instant (fréquence GPU/throttling, contention mémoire, kernels déjà chargés, concurrence
CPU/GPU). Deux configs différentes = tuilage différent, nombre de warps différent, **ordre de réduction
différent** → écart de quelques ulps en bf16/fp16, suffisant pour faire basculer un argmax proche.

**Établi par source primaire** : `triton-lang/triton#9368` documente que `TRITON_CACHE_DIR` seul **ne
garantit pas** la reproductibilité bit-à-bit — même quand les logs indiquent qu'aucun retiming n'a eu
lieu, les sorties peuvent différer ; retirer `@autotune` restaure la stabilité bit-à-bit dans le cas
rapporté. `vllm-project/vllm#58899` documente exactement ce symptôme en production : sortie greedy qui
change entre redémarrages, même avec `VLLM_BATCH_INVARIANT=1`, à cause d'un kernel fusionné q/k-norm+RoPE
qui choisit sa config de réduction par chronométrage.

**Neutralisation** (les 3 modèles convergent, ordre de robustesse décroissante) :
1. Retirer `@triton.autotune` sur les kernels critiques, imposer une config unique et fixe
   (`BLOCK_M`/`num_warps`/`num_stages` codés en dur).
2. Si l'autotuning est gardé : le faire tourner une fois dans un environnement contrôlé, **puis figer et
   partager le cache `TRITON_CACHE_DIR` AVANT le premier appel réel** (préconstruit dans l'image, monté en
   lecture seule, même version Triton/PyTorch/CUDA/driver pour tous les processus).
3. Journaliser via `TRITON_PRINT_AUTOTUNING=1` le choix effectif à chaque démarrage pour vérifier qu'il
   est bien identique entre deux processus.
**Point de vigilance signalé par Luna** : un cache TorchInductor ou un cache propre au moteur peut
masquer/remplacer le cache Triton — vérifier le chemin de cache effectivement utilisé, pas seulement
`TRITON_CACHE_DIR`.

### (b) Heuristique cuBLAS/cuBLASLt et alignement des pointeurs — nuancé, pas une cause démontrée pour l'appel standard

`cublasLtMatmulAlgoGetHeuristic()` prend en entrée des descripteurs de layout/forme et une préférence,
**pas les pointeurs A/B/C/D directement** (doc NVIDIA) — pour un appel direct à formes/layouts/workspace
identiques, rien ne documente que l'heuristique change simplement parce que l'adresse virtuelle a changé
entre deux processus. **Hypothèse trop forte, non établie par les sources.**

Nuance retenue (Luna + Gemma) : l'alignement (`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_*`) fait partie des
critères d'admissibilité d'un algorithme — si **votre code** calcule et transmet un alignement/une
préférence dépendant de l'adresse allouée (pas l'API cuBLASLt elle-même), le jeu d'algorithmes
admissibles peut changer. À vérifier dans le code d'intégration, pas dans cuBLASLt lui-même.

**Neutralisation établie** :
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (ou `:16:8`) — **avant toute création de contexte CUDA**, en tête du
  script/de l'environnement du processus.
- `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark = False`,
  `torch.backends.cudnn.deterministic = True`.
- Pour une garantie plus forte sur cuBLASLt : ne jamais passer `algo = NULL` (sélection implicite à
  chaque appel), demander les candidats une fois, fixer explicitement un `cublasLtMatmulAlgo_t` et le
  réutiliser, avec un workspace préalloué de taille fixe.
- Méthode de diagnostic pratique (Luna) : journaliser par GEMM — M,N,K, strides, type, adresses modulo
  16/32/64/128/256, taille workspace, algo id/tile/stages/split-K — pour trancher si l'algo change vraiment.

### (c) Flash-attention split-KV / `num_splits` — plausible mais non établi comme dépendant de la mémoire libre instantanée

Le partitionnement K/V en splits pour paralléliser la réduction peut changer l'ordre de sommation. Mais
les sources publiques ne démontrent **pas** que l'implémentation standard recalcule `num_splits` à partir
de `cudaMemGetInfo()`/mémoire libre instantanée — la littérature décrit plutôt des seuils statiques
(longueur de séquence, nombre de têtes, nombre de SM). **Hypothèse non établie pour le chemin standard** ;
possible dans un wrapper local/vLLM/xFormers ou un fork — à vérifier dans le code exact du moteur.
L'option `deterministic=True` du dépôt flash-attention concerne le **backward**, pas le choix de
`num_splits` au forward.

**Neutralisation** : pas de variable d'environnement documentée pour figer `num_splits` (les 3 modèles
s'accordent sur ce point). Il faut modifier l'appel/le code pour imposer un `num_splits` fixe (1, ou une
valeur codée en dur), désactiver le chemin split-KV pour isoler la cause, et comparer eager vs éager avec
Nsight Systems pour voir si le nombre de splits utilisé varie réellement entre les deux processus.

### (d) Capture de graphe CUDA dépendante de la mémoire libre — vraie pour les adresses, pas pour l'arithmétique directement

Deux captures neuves peuvent obtenir des adresses différentes (état de l'allocateur, ordre des
allocations). Mais une adresse différente dans un graphe **correctement capturé et rejoué ne change pas
le résultat numérique** — elle ne devient une source de divergence que si elle entraîne un chemin
différent : échec/fallback eager, taille de workspace différente, configuration de kernel différente.

**Neutralisation** : préallouer poids/KV-cache/buffers/workspace avant la capture, warmup complet avant
capture, capturer toujours avec les mêmes buffers statiques, vérifier que la capture réussit et que le
nombre de nœuds du graphe est identique entre les deux processus, désactiver les fallbacks silencieux.

### (e) Ordre d'initialisation, atomics — classique et confirmé, à instrumenter en dernier

Un `atomicAdd` flottant dans un kernel maison (accumulation inter-CTA, split-K/split-KV avec réduction
globale, kernels fusionnés) est une source directe et documentée de non-déterminisme : l'ordre d'arrivée
des CTA dépend de l'ordonnancement matériel, différent d'un lancement à l'autre. Profil « 1 requête sur 5
diverge » compatible avec une race sur un accumulateur partagé. L'ordre d'initialisation (streams,
handles cuBLAS/cuBLASLt, chargement des modules) n'est pas lui-même une source arithmétique — c'est un
déclencheur indirect qui peut faire varier l'état observé par un autotuner.

**Neutralisation** : remplacer `atomicAdd` par une réduction hiérarchique à ordre fixe si possible ; sinon
aucune variable d'environnement ne rend `atomicAdd` déterministe (les 3 modèles s'accordent). Pour le
diagnostic seulement (jamais en production) : `CUDA_LAUNCH_BLOCKING=1` pour sérialiser et isoler une
race/un ordre.

## Désaccord entre modèles — gpt-oss écarté

**gpt-oss invente plusieurs variables d'environnement qui n'existent pas dans aucune documentation
officielle** : `TRITON_DISABLE_AUTO_TUNE`, `TORCH_CUDA_CUBLAS_ALLOW_TF32`,
`TORCH_CUDA_FORCE_DETERMINISTIC`, `CUDA_RANDOM_SEED`, `FLASH_ATTENTION_FORCE_SPLIT_K`, `CUDA_ALLOW_TF32`,
`CUDA_GRAPH_CAPTURE_DEBUG` — et attribue certaines de ces affirmations à des sources qui ne les
contiennent pas (ex. attribue le point vLLM #58899 au blog Thinking Machines). **Non retenu**, même
schéma de confabulation que déjà observé en Q35 sur ce modèle : à ne plus solliciter sans recherche web
activée et vérification croisée systématique des identifiants précis qu'il cite. GPT-5.6 Luna et Gemma 4,
recherche web activée, ont explicitement signalé n'avoir trouvé aucune variable de ce type et l'ont dit.

## Procédure de localisation recommandée (Luna, reprise par Gemma)

Dans cet ordre : (1) désactiver tous les autotuners (Triton, cuBLASLt, kernels maison), config statique ;
(2) `CUBLAS_WORKSPACE_CONFIG=:4096:8` avant lancement ; (3) `torch.use_deterministic_algorithms(True)`
sans `warn_only` pendant le diagnostic ; (4) forcer `num_splits=1` dans le flash-decoding ; (5) désactiver
temporairement les graphes CUDA, comparer eager contre eager ; (6) désactiver un par un les kernels
fusionnés maison, puis flash-attn, puis cuBLASLt ; (7) comparer les tensors après chaque bloc du modèle
entre les deux processus, pas seulement les tokens finaux ; (8) journaliser à chaque lancement : choix
Triton, `num_splits`, algo cuBLASLt, adresses modulo alignement.

**RESTE** : rien en attente de ma part sur Q37 — livré à poste6 et chef.
