# acvram_mojo — étape 0 (à sec) : Mojo/MAX sur sm_120, verdict NUANCÉ (poste4, 01/10 12h19)

Bead : anticitoyen-vram-3yf.2 (P1). Contrat : `revue/moteurs-rust-mojo-23-09.md`. Question posée :
« Mojo/MAX supporte-t-il sm_120 (Blackwell grand public) aujourd'hui ? Version, source primaire,
ce qui manque (MMA FP4, TMA). »

## Verdict

**Ni VRAI ni FAUX au sens simple — compile et tourne, mais sans le chemin accéléré dont acvram
a besoin.** Mojo/MAX s'installe et exécute des noyaux sur sm_120, mais **FP4 et FP8 (donc NVFP4,
le format de poids réel d'acvram) sont verrouillés au sous-ensemble datacenter SM100** et ne
tournent pas sur le Blackwell grand public — confirmé par Modular elle-même, pas par une
supposition.

## Sources primaires (lues directement)

**1. Table de compatibilité GPU officielle** (`docs.modular.com/max/packages`, section « NVIDIA
GPUs ») : `RTX 50XX series | Blackwell` est listée sous **« Known compatible for development »**
(pas sous « Tested for serving » — seul `B200` y figure). Condition logicielle : pilote NVIDIA
≥ 580 (sinon `MODULAR_NVPTX_COMPILER_PATH` vers un `ptxas` système). → **sm_120 est officiellement
dans le périmètre supporté pour le développement**, ce n'est pas un verdict « impossible ».

**2. Forum Modular officiel, fil « Optimized Kernels for Blackwell -- do they work on GB10 »**
(forum.modular.com, 22/06/2026, utilisateur `gchauhan`, réponse d'un salarié Modular
`BradLarson`) — rapport précis sur Blackwell grand public/DGX Spark (sm_121, famille proche de
sm_120, même verrouillage racine) :

| format | GPU Blackwell grand public (sm_12x) |
|---|---|
| `bfloat16` | ✅ seul chemin GPU qui fonctionne |
| `float8_e4m3fn` (fp8) | ❌ **verrouillé SM100** |
| `float4_e2m1fnx2` (fp4/NVFP4) | ❌ **verrouillé SM100** |
| `q4_k`/`q6_k` (GGUF) | ❌ CPU seulement |

Plus : l'attention Flash **ne passe pas par le chemin optimisé** sur ce matériel (repli générique
lent). Le salarié Modular confirme que l'équipe **« looking to prioritize support for consumer
GPUs »** — aveu que ce n'est pas encore priorisé au 22/06/2026.

**3. Confirmation indépendante, même mécanisme, autre projet** (issue GitHub
`vllm-project/vllm#31085`, « Add SM120 support for native NVFP4 MoE kernels ») : la sélection de
backend MXFP4/NVFP4 de vLLM teste littéralement `is_device_capability_family(100)` — sm_120
`(12, 0)` ne matche pas cette famille, le verrouillage SM100 est donc un motif architectural
réel et répété (mma.sync sm_120 contre `tcgen05.mma` sm_100 — deux jeux d'instructions Tensor
Core différents), pas une simple absence de drapeau de compilation chez Modular seul.

## Ce qui manque précisément

* **MMA FP4/FP8** : les noyaux matmul Blackwell optimisés (gemm NVFP4, DeepGEMM, collectives
  CUTLASS SM100, Flash Attention WGMMA) ciblent `tcgen05.mma` (SM100) ; sm_120 n'a que les
  instructions `mma.sync` héritées (style SM80) — pas un sous-ensemble de SM100, une architecture
  Tensor Core différente. Résultat : les noyaux écrits pour SM100 échouent à la compilation ou
  plantent à l'exécution sur sm_120.
* **TMA** (Tensor Memory Accelerator) : non vérifié directement dans nos sources Mojo — l'écart
  FP4/FP8 déjà confirmé suffit à répondre à la question posée ; à vérifier séparément si l'écart
  FP4 est un jour comblé.
* Version Mojo/MAX exacte testée par le rapport forum : non précisée dans le fil lu ; daté
  22/06/2026, MAX en développement continu depuis.

## Conséquence pour acvram_mojo

Les poids acvram réels sont **NVFP4/int8** (`acvram-gemma-4-31b-it-nvfp4-…`, manifeste cité par
poste5 en 3yf.1). Un squelette Mojo aujourd'hui ne pourrait décoder qu'en **bf16** — ni le format
de poids servi, ni une comparaison valable au moteur Python acvram (contrat § 3 : identité au
bit ou KL ≤ 0,74 contre LA MÊME référence quantifiée). Construire le squelette maintenant
testerait un modèle différent de celui que (a)/(b) du contrat visent à isoler.

## Reste

Pas de squelette construit à ce stade — **décision à chef** : (1) squelette bf16 quand même,
utile seulement pour mesurer le langage hôte sur un modèle non quantifié (hors du périmètre du
contrat § 2) ; (2) attendre que Modular ouvre FP4/FP8 sur sm_120 avant tout squelette (date
inconnue) ; (3) clore 3yf.2 en « verdict publié, non retenu » et réaffecter le budget à
acvram_rust/autre pièce. Aucune carte utilisée (recherche à sec, comme demandé).

**RESTE** : rien en cours. Décision suivante : chef.
