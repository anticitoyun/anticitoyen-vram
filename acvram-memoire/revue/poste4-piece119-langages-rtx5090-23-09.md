# Pièce 119 — langages de programmation et RTX 5090 (demande utilisateur), à sec

Question posée mot pour mot aux trois modèles duck.ai (raisonnement) : « pour un moteur d'inférence efficace
quels languages de programmation pour une rtx5090 fait la différence ? ». Puis recherche profonde (WebSearch,
WebFetch, GitHub, arXiv) sur CUDA C++, CUTLASS/CuTe DSL, Triton, Mojo, TileLang, ThunderKittens, PTX/SASS à la
main, Rust (cudarc), Python — chiffres vérifiables sur sm_120/Blackwell, liens à l'appui. Aucune décision prise
ici : nommer ce qui gagnerait chez nous, où, à quel coût — la décision reste à chef/poste7.

## 1. Réponses duck.ai (3/3, mode raisonnement)

**Gemma 4 31B** (9s) : C++ reste le choix dominant (TensorRT, llama.cpp) pour la gestion mémoire fine ; CUDA C++
pour les noyaux exploitant FP8/FP4 ; Triton pour approcher CUDA C++ sans sa complexité, utile contre les
goulots de bande passante GDDR7 ; Rust en progression côté hôte (sécurité mémoire, buffers massifs) ; Mojo cité
comme alternative prometteuse (Python + performance C++). Conclusion : la différence se joue sur la capacité à
descendre bas dans la pile logicielle, pas sur un langage unique.

**GPT-5.6 Luna** (14s, cite `docs.nvidia.com`/`forums.developer.nvidia.com`) : « le langage seul fait peu de
différence » — la performance vient de TensorRT, CUDA Graphs, la précision (FP16/FP8/INT8), le batch et les
copies CPU↔GPU. Table : C++ + CUDA = maximal (latence minimale, zéro copie) ; Python = très proche via
TensorRT/PyTorch (prototypage, LLM, orchestration) ; Triton = très élevé pour des noyaux personnalisés sans
tout réécrire ; Rust = bon via bindings (`rust-cuda`, `tch-rs`) pour la sûreté système. Conclusion pratique :
« Python lance des kernels qui s'exécutent sur le GPU ; si chaque appel traite un batch suffisamment important,
le surcoût Python est souvent négligeable » — pénalisant seulement à beaucoup de petits appels synchrones. LLM
local : « Python avec TensorRT-LLM, vLLM ou SGLang [...] réécrire l'application en C++ n'apporte généralement
pas un gain important ».

**gpt-oss 120B** (1s, réponse partiellement en chinois malgré la question en français — anomalie de langue à
noter, sans lien ni chiffre vérifiable) : C/C++ pour la latence/débit minimal (CUDA/cuDNN) ; Python via
PyTorch/TensorFlow/JAX, cœur toujours en C/C++ ; Rust via `rust-cuda`/`tch-rs`, sûreté mémoire ; Julia
(`CUDA.jl`/`Flux.jl`) génère du PTX efficace. Conclusion identique aux deux autres : C/C++ + Python est la
combinaison pratique dominante.

**Accord des trois** : le langage seul ne fait pas la différence ; ce qui compte est d'atteindre les noyaux
Tensor Core natifs et de traiter des lots assez gros pour amortir l'orchestration Python. Aucun chiffre sourcé
n'a été donné par gpt-oss ; Luna cite deux pages NVIDIA génériques (support matrix TensorRT, forums) sans
chiffre isolé pour sm_120 précisément.

## 2. Recherche profonde, par outil — chiffres vérifiables sm_120/Blackwell

### CUDA C++ à la main (sans DSL)

* **`waynehacking8/blackwell-tensorcore-kernels`** (GitHub, README à jour) — méthode : échelle de noyaux naïf →
  GEMM à mémoire partagée → WMMA → **`mma.sync` brut**, comparés à la même précision que `cublasGemmEx` (FP16
  in / FP32 acc, Tensor Cores). Résultat mesuré sur RTX PRO 6000 (sm_120) : le noyau **`mma.sync` brut atteint
  106 % de cuBLAS-TC (243 contre 229 TFLOP/s à 8192³)**, battant le noyau `cutlass_80` que cuBLAS lui-même
  dispatche sur cette carte. Le même noyau écrit avec l'API WMMA (toujours du CUDA C++, mais via un wrapper)
  ne fait que **45 % de cuBLAS-TC sur sm_120** — l'écart n'est pas le langage, c'est l'instruction émise
  (`mma.sync` natif contre l'abstraction WMMA qui ne l'exploite pas pleinement).
* **`gau-nernst.github.io/fa-5090`** (blog, code source lié) — Flash Attention écrite à la main en CUDA C++
  pour la RTX 5090, b=1, 8 têtes, 4096 requête / 8192 KV, 400 W, CUDA 12.9 (SOL BF16 théorique 209,5 TFLOPS) :
  v1 (basique) 68,2 % du SOL → v5 (pipelining amélioré) **94,39 %**, contre `F.sdpa()` (backend Flash Attention
  PyTorch) 89,13 %, `F.sdpa()` (cuDNN) **97,19 %**, la bibliothèque `flash-attn` 90,97 %. **L'auteur note
  explicitement que Triton n'a pas de MMA MXFP8/NVFP4 pour sm_120** — raison donnée pour écrire en CUDA C++
  plutôt qu'en Triton pour ce projet précis.
* **Conclusion CUDA C++ à la main** : plafond le plus haut atteignable sur sm_120, mais seulement si l'on émet
  les instructions natives (`mma.sync`, `cp.async.bulk`/TMA) — un simple portage vers une API haut niveau du
  même langage (WMMA) ne suffit pas.

### CUTLASS / CuTe DSL (Python-embarqué, génère du SASS via nvcc/CUDA 13)

* **Colfax Research**, série NVFP4 blockscaled GEMM sur RTX PRO 6000 (sm120), CuTe DSL (`nvidia-cutlass-dsl`
  4.6.0, Python 3.13.13, PyTorch 2.12.1) : version de base 73 % d'utilisation à 8k (1476 TFLOP/s, pic carte
  2015,2 TFLOP/s FP4), ≈93 % de cuBLAS 13.6 en moyenne sur cinq tailles. Après optimisations (swizzling,
  micro-opts) : **jusqu'à 1666 TFLOP/s à 16k, 83 % d'utilisation**, gains de +4 % (8k) à +40 % (32k) selon la
  taille. Code publié : `github.com/ColfaxResearch/cfx-article-src/tree/master/sm120_nvfp4_gemms`.
* **`VincentKaufmann/fp4-cuda-kernel`** (GitHub) — noyau FP4 custom pour SM120/121 sur CUTLASS 3.8 :
  **85-129 TFLOPS**, ×1,4-2,4 plus rapide que BF16 aux tailles de lot d'inférence, ×4 de mémoire économisée.
* **`NVIDIA/cutlass` issue #3096 et `flashinfer-ai/flashinfer` issue #2723** — la MoE NVFP4 groupée de CUTLASS
  produisait une sortie invalide sur SM120 (bug daté, corrigé via patches FlashInfer + `compute_120f`,
  CUDA 13.0 — **39 tok/s en FP4 natif** une fois corrigé) : CuTe DSL/CUTLASS sur sm_120 est puissant mais
  moins mature que sur les cartes datacenter (B200/GB200), avec des bogues de géométrie encore ouverts en 2026.
* **Conclusion CUTLASS/CuTe DSL** : proche du plafond matériel (83-93 % de cuBLAS) accessible sans écrire de
  SASS à la main, mais la couverture sm_120 spécifique (par opposition à sm_100/B200) est plus récente et
  moins éprouvée — bogues de géométrie MoE groupée documentés.

### Triton

* **Limite nommée et vérifiable** : `gau-nernst.github.io/fa-5090` déclare explicitement l'absence de MMA
  MXFP8/NVFP4 en Triton pour sm_120 (raison de son choix du CUDA C++ pour ce projet).
* Aucun chiffre de `% de pic` Triton sur sm_120 FP4 spécifiquement trouvé et vérifiable dans cette recherche
  (les pages générales RTX 5090/Blackwell ne détaillent pas Triton isolément) — à consigner comme trou, pas
  comme un chiffre reconstruit (REGLES point 10).
* Ce que les moteurs qui gagnent utilisent : **vLLM** écrit ses noyaux de décodage (`unified_attention`, pièce
  93/Q18 de ce projet) en Triton pour la portabilité, mais garde du CUDA C++/CUTLASS pour les GEMM quantifiées
  critiques (Marlin, FlashInfer) — Triton n'est donc pas le langage du chemin le plus chaud chez eux non plus.

### Mojo

* **Modular @ GTC 2026** (blog officiel) : portage du noyau conv2d Blackwell de CUTLASS depuis CUDA C++ vers
  Mojo — **130,7 TFLOPS sur B200, égalant le débit CUTLASS, en ~770 lignes de Mojo contre ~3000 en CUTLASS**.
  Chiffre concret et vérifiable, mais mesuré sur **B200 (datacenter), pas sur RTX 5090/sm_120** — à ne pas
  transposer sans le dire.
* MAX (moteur Modular) revendique ≈4,5× le débit de PyTorch+HuggingFace pour un LLM 7B à b=64 sur **un H100**
  (finance-orienté) — même réserve : pas une mesure sm_120.
* Mojo 1.0 publié le 11/08/2026, compilateur Apache-2.0 le 18/08/2026 — écosystème jeune, aucun chiffre RTX
  5090 trouvé dans cette recherche.

### TileLang

* Papier officiel (`arxiv.org/pdf/2504.17577`) : benchmarks sur H100, pas de chiffre sm_120/RTX 5090 trouvé.
* `arxiv.org/abs/2604.23466` (« Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs ») compare
  cuBLAS/Triton/WMMA/SIMT sur Hopper ET Blackwell — pertinent en principe mais le contenu détaillé n'a pas été
  extrait ici (à lire si le sujet devient un chantier réel).
* Pas de chiffre sm_120 vérifiable trouvé pour TileLang spécifiquement — trou nommé, non chiffré.

### ThunderKittens

* ThunderKittens 2.0 (11/01/2026, Hazy Research/Together AI) : support Blackwell complet, MXFP8/NVFP4 ; noyaux
  BF16/FP8 « à ou proche de la vitesse cuBLAS », jusqu'à 2× plus rapide que cuBLAS sur H100 ; attention proche
  de cuDNN sur B200, jusqu'à 2× FA3 sur H100. **Ces chiffres sont H100/B200, pas sm_120/RTX 5090** — le dépôt
  supporte SM120 dans sa suite de tests mais aucun chiffre isolé sm_120 n'a été trouvé et vérifié ici.
* Référencé comme ligne de comparaison (± 12 % de ses chiffres publiés) dans le banc de
  `waynehacking8/blackwell-tensorcore-kernels` ci-dessus — cohérence croisée, sans chiffre sm_120 propre.

### PTX/SASS à la main

* **`kacper-daftcode/vLLM-Moet`** (déjà relevé pièce 119-antérieure/recherche RTX 5090) — noyaux SASS SM120
  écrits à la main pour des experts MoE 2 bits + cache delta FP4 ; le README/l'article `starlog.is` cité
  affirment qu'**aucun de ces noyaux ne peut être obtenu par du CUDA stock sur sm_120** (« the instructions
  compile, the kernels don't ») — SASS à la main est présenté comme la SEULE voie pour ce cas précis
  (2 bits + delta FP4), pas une optimisation parmi d'autres.
* **`zolotukhin.ai/blog/2026-06-12-...`** (« RTX 5090 LLM Decode: MoE & Gemma in ZINC's CUDA Backend ») —
  cité par la recherche comme traitant quatre goulots d'étranglement d'un moteur CUDA maison sur 5090 ; contenu
  détaillé non extrait ici (lien à suivre si utile).
* Le résultat `waynehacking8` ci-dessus (106 % de cuBLAS-TC) est déjà obtenu en `mma.sync` (PTX intrinsèque en
  CUDA C++), sans descendre au SASS brut — **le SASS à la main n'est nécessaire que quand l'instruction visée
  n'a pas d'intrinsèque PTX exposée par nvcc** (cas MoE 2 bits de vLLM-Moet), pas pour un GEMM Tensor Core
  standard.

### Rust (cudarc et équivalents)

* **`avifenesh/memra`** (GitHub) — moteur Rust + CUDA pour RTX PRO 6000/RTX 5090 (safetensors + GGUF, API
  OpenAI, décodage spéculatif) : **140 tok/s sur RTX PRO 6000, 75 tok/s sur « RTX 5090 Laptop »** (pas la carte
  de bureau) — chiffre trouvé mais géométrie/modèle non précisés dans l'extrait, à vérifier avant de le citer
  ailleurs comme référence Rust.
* Pas d'autre chiffre Rust/`cudarc` isolé et vérifiable sur sm_120 trouvé dans cette recherche.
* Aucun moteur dominant (vLLM, llama.cpp, TRT-LLM, SGLang, b12x, NInfer) n'a son chemin chaud en Rust — Rust
  sert de couche hôte/serveur (mémoire sûre, buffers), jamais des noyaux Tensor Core, dans tout ce qui a été vu
  ici et dans la pièce 111 (`nibor1896/crow-nest`, Rust + CUDA, mêmes rôles : hôte + kernels maison en CUDA,
  pas de noyau Tensor Core en Rust pur).

### Python (couche d'orchestration, pas d'exécution)

* Confirmé par les trois réponses duck.ai et par la recherche : Python n'exécute jamais le calcul chaud —
  il lance des noyaux CUDA C++/Triton/CUTLASS. Le surcoût de lancement Python/CPU (5-10 µs par noyau, des
  centaines de noyaux par passe) peut représenter 20-40 % du temps d'inférence à b=1 sans CUDA Graphs ; les
  CUDA Graphs éliminent ce surcoût côté hôte (~50-100 µs/jeton) pour un gain mesuré de 10-20 % au petit lot,
  négligeable au lot large où le calcul domine. **acvram utilise déjà les CUDA Graphs** (docs/ARCHITECTURE.md)
  — ce levier est déjà pris chez nous.

## 3. Ce que les moteurs qui gagnent utilisent réellement

| Moteur | Langage du chemin chaud | Rôle de Python | Source |
|---|---|---|---|
| **NInfer** (pièce 103) | C++ + CUDA (`.cpp`/`.cu`, `ninfer::ops::`), CMake, Apache-2.0 | aucun observé (bibliothèque C++ pure) | clone lu directement, pièce 103 |
| **vLLM** | Triton (décodage, `unified_attention`) + CUDA C++/CUTLASS pour les GEMM critiques (Marlin, FlashInfer) | orchestration, scheduler, API | pièce 93/Q18 (ce projet) |
| **llama.cpp** | C/C++ pur, aucun Python au chemin chaud | build/scripts annexes seulement | notoriété du projet (GGML/GGUF, cf. Gemma ci-dessus) |
| **TRT-LLM / TensorRT** | C++ (moteur TensorRT compilé), Python pour la construction du graphe et l'API | construction du plan, orchestration | `docs.nvidia.com` (support matrix), Luna |
| **b12x** (pièce 111) | CuTe DSL (Python-embarqué, compile en SASS via `nvidia-cutlass-dsl`) + Triton pour l'auxiliaire | `PreparationSession` compile les noyaux, sinon aucun calcul | pièce 111 (lecture directe du dépôt) |
| **vLLM-Moet** | SASS SM120 écrit à la main, patché sur vLLM | héberge le reste de vLLM (Python) | ci-dessus |

Constante partagée par tous : **aucun moteur qui gagne n'exécute son calcul chaud en Python** ; ils diffèrent
sur CE qui remplace Python (CUDA C++ à la main, CuTe DSL, Triton, ou SASS à la main pour les cas que rien
d'autre ne peut exprimer).

## 4. Pour acvram (Python + CUDA C++ maison + port Marlin) — ce qui gagnerait, où, à quel coût

**Aucune décision prise ici** — trois observations nommées, chacune avec sa mesure de vérification, à trancher
par qui de droit :

1. **Notre chemin chaud est déjà dans la bonne catégorie.** acvram écrit ses noyaux critiques en CUDA C++
   (`acvram_kernels.cu`) et Triton (`route_prep`, `gemm_etroit`) — la même famille que vLLM/llama.cpp, pas une
   couche Python interprétée sur le calcul. Le levier « changer de langage » n'est donc PAS le premier
   candidat ; le levier « quelle instruction chaque noyau émet-il » l'est (cf. § 2, `mma.sync` contre WMMA :
   45 % contre 106 % du même plafond, dans le même langage).
2. **Piste nommée par le § 2 : vérifier que nos GEMM Tensor Core émettent bien `mma.sync`/TMA natif, pas une
   API de plus haut niveau qui les enterre.** Mesure la moins chère : `nvcc -Xptxas -v` + `cuobjdump -sass`
   sur un noyau GEMM existant (`gemm_etroit`, `gemv_marlin`), comparer aux instructions citées par
   `waynehacking8` — coût : une lecture de SASS, aucune carte, < 30 min.
3. **CuTe DSL comme piste d'évolution, pas de remplacement.** Le résultat Colfax (83-93 % de cuBLAS en CuTe
   DSL, Python-embarqué) montre qu'on peut approcher le plafond CUDA C++ à la main SANS quitter un flux
   Python-adjacent — pertinent si un futur noyau GEMM/MoE demande plus d'itération que le CUDA C++ brut n'en
   permet dans le budget de la campagne. Coût : dépendance à `nvidia-cutlass-dsl` (CUDA 13, encore jeune sur
   sm_120 — bogues MoE groupée documentés § 2), à peser contre le bénéfice avant tout chantier.
4. **SASS à la main : seulement si une instruction manque, jamais par défaut.** Le seul cas documenté où le
   SASS à la main est présenté comme NÉCESSAIRE (pas seulement plus rapide) est le MoE 2 bits + delta FP4 de
   vLLM-Moet, une géométrie que CUDA stock ne compile pas correctement sur sm_120. Tant qu'aucun de nos noyaux
   n'a ce problème nommé, ce levier reste hors sujet.
5. **Triton : lacune connue, nommée, pas à corriger par nous.** L'absence de MMA MXFP8/NVFP4 en Triton sur
   sm_120 (constatée par un tiers indépendant) explique pourquoi nos noyaux Triton (`route_prep`, `gemm_etroit`)
   restent probablement en formats plus anciens pour leurs parties MMA — à vérifier si un de nos noyaux Triton
   tente une MMA FP4/FP8 directe, auquel cas la limite pourrait déjà nous toucher sans qu'on l'ait nommée.
6. **Python (orchestration) : déjà couvert par les CUDA Graphs.** Le gain générique « moins de Python » (10-20 %
   au petit lot) est déjà capté chez nous par les graphes CUDA existants — pas un levier neuf à chiffrer.

## Sources principales

Duck.ai (3 réponses complètes) ; GitHub : `waynehacking8/blackwell-tensorcore-kernels`,
`VincentKaufmann/fp4-cuda-kernel`, `kacper-daftcode/vLLM-Moet`, `avifenesh/memra`, `NVIDIA/cutlass#3096`,
`flashinfer-ai/flashinfer#2723`, `ColfaxResearch/cfx-article-src` ; blogs : `gau-nernst.github.io/fa-5090`,
`research.colfax-intl.com` (×2), `hazyresearch.stanford.edu`, `modular.com/blog` ; arXiv :
`2504.17577` (TileLang), `2604.23466` (CUDA Tile Hopper/Blackwell).
