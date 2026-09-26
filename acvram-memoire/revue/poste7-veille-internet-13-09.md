# poste7 — veille internet 13/09/2026 : ce qui peut servir acvram

Demande de l'utilisateur, réponse à chef. Méthode : recherche web (≈40 requêtes),
puis **lecture directe** des abstracts arXiv, README, issues GitHub et bancs
publiés (scripts, pas de résumé intermédiaire) pour chaque chiffre cité ci-dessous.
Ce qui n'a été vu qu'à travers un résumé de moteur de recherche est marqué *(résumé)*.
duck.ai **non consulté** dans cette passe (règle §1 : à faire par les sessions sur
les questions ouvertes en §7). Rien lancé sur la carte. Déjà connu du dépôt (grep
`github.com/` dans docs/, acvram-memoire/, revue/) : llama.cpp, vLLM, TRT-LLM,
SGLang, FlashInfer, CUTLASS, ModelOpt, colibrì, ggrun, ktransformers,
ik_llama.cpp, lna-lab, JohnTDI, mistral.rs, GPTQModel. **Nouveau** : tout le reste.

## 0. Les huit faits qui changent une décision chez nous

1. **Le bridage 400 W est probablement inerte au décodage, et l'horloge SM est le
   vrai levier** (arXiv 2605.11999, mai 2026) : sur H200, le décodage tire
   137-300 W sur 700 W, aucun cap ne se déclenche ; le verrouillage d'horloge SM
   « Pareto-domine » le cap et **récupère jusqu'à 32 % de l'énergie de décodage**
   à débit quasi égal ; le pilote clampe silencieusement `--lock-gpu-clocks` ≥ 1830
   à ≈ 1830 MHz. Confirmé indépendamment par 2501.08219 : décodage = 77-91 % du
   temps, insensible à la fréquence, 2842 → 180 MHz = **−42 % d'énergie pour
   +1-6 % de latence**. Coût chez nous : une ligne `nvidia-smi -lgc`, mesure par
   `energie.py`.
2. **llama.cpp est la référence en joules, pas vLLM** (HF, discussion
   Qwen3.8-27B #132, août 2026, un relevé) : vLLM NVFP4 sur 5090 = 123 t/s à
   **380-520 W** (3,1-4,2 J/jeton) ; llama.cpp Q4_K_M = 78,9 t/s à **120 W**
   (1,5 J/jeton). Le plus rapide est 2-2,8× plus gourmand par jeton.
3. **Le prior de routage sur notre modèle existe** (arXiv 2608.18261, août 2026,
   mesuré sur **Qwen3-30B-A3B**) : réutilisation entre jetons adjacents = 2,0× le
   hasard ; **95 % du trafic passe par 52,5 % des experts** ; **une LRU de 13,4 % des
   experts sert 66 % des requêtes** ; le routage « code » est presque orthogonal à
   prose/maths. Ma prédiction d'hier (h_pin(64) = 0,72) est donc probablement
   **trop basse** sur un seul domaine (prior 0,85-0,95 à C = E/2) et juste sur un
   mélange. Entraîner le routeur pour la localité **échoue** au seuil ≤ 1 % de PPL
   (résultat négatif pré-enregistré).
4. **Comment une évaluation de cache d'experts ment** (arXiv 2608.07911, v4 du
   25/08/2026, déjà cité dans FDR:2396) : rejeu « par accès » au lieu de
   « par événement fusionné » gonfle les politiques de récence de **27-29 %** et
   inverse le classement ; contamination par gabarit de prompt déplace l'effet de
   19-32 points ; il faut publier **l'union d'experts par pas rapportée à la
   capacité par couche**. C'est mot pour mot le protocole M1 de
   `poste7-cache-experts-13-09.md` §5 — à appliquer tel quel.
5. **Une synchronisation hôte par couche tue un cache même à 97-99 % de succès**
   (llama.cpp discussion #24528, juin 2026, 33 commentaires) : le slot-pool Metal
   était 2× plus lent que vanilla à 97-99 % de succès, « purement par les points
   de synchronisation par couche » ; l'offload forcé sans résidence = régression
   3×. Ça valide la table d'adresses sans décision hôte (§4 de ma note) et **tue
   l'option B** (rassemblement décidé côté hôte).
6. **SM120 = `mma.sync` d'Ampère + MMA FP4 bloc-échelle native, 99 Ko de shared,
   ni `tcgen05`, ni multicast** (lna-lab README ; prouvé chez nous par poste4,
   `outils/test_mxf4_sm120.cu`). Conséquences mesurées ailleurs : le GEMM groupé
   CUTLASS bloc-échelle rendait du **garbage** sur SM120 jusqu'en mars 2026
   (cutlass #3096, flashinfer #2723 ; correctif = `compute_120f` + patches) ; au
   décodage, **Marlin W4A16 (46-50 t/s) bat le FP4 natif groupé (14,6 → 39 t/s)**
   sur Qwen3.5-397B/4×PRO 6000. Même conclusion que notre banc prefill MoE
   (ratio 1,00) : GEMV W4A16 au décodage, FP4 natif au prefill seulement.
7. **Un moteur natif consommateur existe déjà dans notre niche** : SparkInfer
   (gittensor-ai-lab, C++/CUDA, binaire 2,5 Mo, « Blackwell-native ») :
   Qwen3.8-27B dense NVFP4 sur 5090 = **95,7 t/s** décodage, 6 900-14 400 t/s
   prefill, brouillon DSpark 1,47× (16k) à 4,01× (4k), agrégé 345 t/s à 8-16
   concurrents contre vLLM 0.28 546 t/s. Et **FreeToken** (Berkeley/MIT,
   Apache-2.0, arXiv 2608.16157) : Qwen3.6-35B-A3B **avec experts hors VRAM** =
   **77-83 t/s sur 5090**, 1,8-2,3× le meilleur témoin, poids bit-identiques ;
   PCIe mesuré 52,7 Go/s sur leur 5090 x16.
8. **La bande PCIe de leur 5090 x16 = 52,7 Go/s** (FreeToken, mesuré) : notre
   `MATERIEL.md:233` (52,8) est un relevé x16 ; notre `FDR:2403` (18,7) un relevé
   x8 négocié. Les deux sont vrais ; **la question M0 est « à combien sommes-nous
   négociés aujourd'hui »**, `lspci -vv | grep LnkSta`.

## 1. Concurrents sur RTX 5090, chiffres publiés (régime nommé)

| moteur | modèle / format | régime | décodage | prefill | source, date |
|---|---|---|---|---|---|
| **acvram** | Coder-30B nvfp4 | b=1 / b=8 | 174-201 / 509 agr. | 6 420 j/s (4k) | `FEUILLE-DE-ROUTE.md:734,1760,1800` |
| llama.cpp | Coder-30B Q4_K_M | b=1 | 182 | — | willitrunai *(résumé)* |
| llama.cpp | Qwen3.5-35B Q4_K_M | b=1, tg128 | **214** | 4 151 | zenn (toki_mwc), avr. 2026 |
| llama.cpp | Qwen3.6-35B Q4_K_M / Q5_K_S | b=1 | 183 / 153 → **180** (b8870) | 2 892 | zenn, avr.-juin 2026 |
| llama.cpp | Qwen3.6-35B Q4_K_XL | b=1 | **207** ; **490** avec n-gram k4v256 sur édition de code | 6 700 (25k) | note.com (unco3), 1/06/2026 |
| llama.cpp | Qwen3.6-35B MXFP4 | b=1 | +9 % seulement | — | note.com |
| vLLM | Qwen3.6-35B NVFP4 (lna-lab patches) | b=1 | **175** (5090), 168 (PRO 6000) | — | lna-lab README |
| vLLM | Qwen3.6-35B (AEON, DFlash + Marlin INT4) | b=1 | **297** ; 1,44× llama.cpp sans spéculation | — | note.com 6/06/2026 *(résumé)* |
| vLLM | Qwen3.5-35B GPTQ-Int4 | b=1 | 194-197 | — | HF Qwen #3 *(résumé)* |
| vLLM | Coder-30B AWQ | agrégé | 556 (faible conc.) ; **1 157** à MCR=16, TPOT 7 ms | — | cloudrift, 2026 |
| SGLang | Coder-30B AWQ | agrégé | 208 (2,7× sous vLLM) | — | cloudrift |
| TRT-LLM | Qwen3-30B-A3B NVFP4 | b=1 | 135, TTFT 15 ms, 24,1 Go | — | HF #24 (JohnTDI), nov. 2025 |
| vLLM | Qwen3.5-27B dense NVFP4 | b=1 | 80 (Marlin) | 4 016 | aliez-ren README |
| SparkInfer | Qwen3.8-27B dense NVFP4 | b=1 | 95,7 ; 130 avec DSpark (16k) | 6 942-14 364 | gittensor README |
| FreeToken | Qwen3.6-35B, **experts hors VRAM** | b=1 | **77-83** | 6,7k (16k, x16) | 2608.16157 + revue Zhou |
| llama.cpp | Qwen3.6-35B sur **3090** | b=1 | 140 (tout VRAM), 89 (10 couches FFN sur CPU) | 3 300 / 1 100 | gilesthomas, juil. 2026 |

Lecture : à b = 1 nous sommes **à parité avec llama.cpp** (174-201 contre 180-214
sur des 35B-A3B voisins) et **sous vLLM en agrégé** (509 à b=8 contre 1 157 à
MCR 16 — régimes différents, à apparier). Le levier que tous exploitent et que
nous n'avons pas en service : **la spéculation** (×1,4-4,3 selon la tâche).

## 2. Ce qui sert, par thème — gain / coût / mesure qui tue

### 2.1 Énergie (objectif joules)

| geste | gain attendu | coût | tue | source |
|---|---|---|---|---|
| **Verrouiller l'horloge SM en décodage** au lieu du bridage 400 W ; horloge haute au prefill (« DVFS par phase ») | −20-30 % J/jeton décodage, débit −≤3 % | 0 (`nvidia-smi -lgc`, script par phase) | `energie.py` : J/jeton −<10 % ou débit −>5 % | 2605.11999, 2501.08219 |
| **Taxe de stationnement des contextes CUDA** : chaque contexte au repos coûte +26-66 W (66 W sur GDDR6), VRAM allouée sans effet (<0,02 W/Go) | nos 3 services = 3 contextes → ~50-150 W permanents ; consolider 8082+8083 en un processus | faible | delta < 10 W au wattmètre NVML | 2605.23918 |
| Régime de mesure J/jeton | énergie par requête ≠ par jeton (7,46 → 0,72 J/jeton de 10 à 512 jetons) : publier les deux | 0 | — | 2608.28044, ML.ENERGY 2505.06371 |
| Points de comparaison consommateur | 0,56-4 J/jeton sur 4060 Ti (1-7B, Ollama) ; llama.cpp 5090 1,5 J/jeton (§0.2) | 0 | — | 2608.00008, HF #132 |

### 2.2 Cache d'experts et exil (suite de `poste7-cache-experts-13-09.md`)

| geste | gain attendu | coût | tue | source |
|---|---|---|---|---|
| **M1 avec le protocole 2608.07911** : rejeu par événement fusionné, moitié apprentissage / moitié évaluation, gabarits de prompts variés, union d'experts par pas ÷ capacité par couche | prior : h_pin(E/2) = 0,85-0,95 sur code seul ; 0,66 à 13,4 % en LRU | 1 h carte | Δh < 0,10 (seuil déjà posé) | 2608.07911, 2608.18261 |
| **Contre-trace llama.cpp** : `llama-moe-trace` (~120 lignes dans l'eval-callback, `ffn_moe_topk`) sur le **même GGUF** → comparer les listes d'experts routés entre moteurs | contrôle croisé du routage (règle 5 : comparer les listes) | 1 h | désaccord > 2 % des jetons = bogue de routage chez l'un | 2608.18261 §4 |
| **Exécuter les experts manqués sur le CPU** (politique q* de FreeToken : répartir les ratés entre transfert PCIe et calcul CPU selon les bandes **mesurées**) | à x8 (18,7 Go/s) le CPU à 71 Go/s DDR bat le PCIe **×3,8** par octet ; à x16 ×1,3 ; 8 experts × 2,66 Mo ≈ 0,35 ms/couche prédit contre 1,14 ms | vérifier que `mlp_exec=cpu` (tiering.py:655) est câblé pour MoE dans le moteur | > 1 ms/couche sur i9-14900K (pas d'AMX : les chiffres KTransformers/CoX-MoE ne se transportent pas) | 2608.16157, 2605.17889 |
| **Zéro synchronisation hôte par couche** — confirmé indispensable | — | — | toute variante avec sync par couche : 2× plus lent même à 97-99 % de succès | llama.cpp #24528 |
| Table id → slot après le top-k (**même geste que ma table d'adresses**) : llama.cpp PR #25294 (ouvert, juil. 2026) : cache par couche de `n_slots` experts, op de remappage, O_DIRECT, prefill par vagues | lecture de leur implémentation avant d'écrire la nôtre | 1 h | — | PR #25294 |
| Double tampon pleine couche au prefill : FreeToken mesure **−19 %** sans second tampon, à x16 et modèle presque entièrement exilé | **correction de ma note d'hier** (« ≤ 5 % ») : ≤ 19 % du prefill des couches exilées à x16, ~0 à x8, 0 au décodage | moyen | prefill 4k, 12 couches exilées : gain < 5 % | 2608.16157 |
| Page cache noyau comme tier d'experts (MGLRU + balloon surestime le trafic 2×) | rien à faire ; **ne pas** mesurer un exil avec un balloon | 0 | — | 2608.12103 |
| Résidence par manifeste de saillance (`expertpin`, MIT, ik_llama.cpp, `--resident-experts K`, mmap/mincore/madvise) | alternative au profil d'usage : un ordre d'experts calculé hors ligne (REAP) | lecture | — | seppegadeyne/expertpin |

### 2.3 Noyaux SM120 (format NVFP4 = le nôtre)

| ressource | ce que c'est | pour nous | source |
|---|---|---|---|
| **TileLang `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`** + `maint/gemm/gemm_sm120/{benchmark, correctness_evaluation_nvf4_vs_cutlass, cutlass_nvf4_ref.cu, tilelang_nvfp4_quantizer.py}` (30/07/2026, `T.mma_gemm_blockscaled`) | GEMM NVFP4 bloc-échelle **sm120 natif** en DSL, avec témoin CUTLASS et quantiseur | chemin court pour le bead W4A4 (brd) : notre format exact, testé contre CUTLASS ; **le prefill MoE n'est pas borné par le GEMM** (banc 12/09, ratio 1,00) → priorité basse tant que ce constat tient | tile-ai/tilelang |
| Colfax « Optimizing an NVFP4 blockscaled GEMM on RTX PRO 6000 (SM120) » : 1 476 → **1 666 TFLOP/s = 83 % du pic 2 015** en CuTe DSL ; horloge tombe 2,43 → 2,15 GHz aux grandes formes (~1 782 réels) | référence de plafond et d'échelle de progression (+29 % à 2k, +40 % à 32k) | borne pour notre GEMM prefill ; poste4 a 2 060 TFLOPS en micro-banc | research.colfax-intl.com |
| CUTLASS `examples/` : `79_blackwell_geforce_gemm`, `87_blackwell_geforce_gemm_blockwise`, **`91_fp4_gemv`**, `92_blackwell_moe_gemm`, `93_blackwell_low_latency_gqa` | 91 = un GEMV FP4 (notre régime décodage), 92 = GEMM MoE, 93 = attention GQA basse latence | **à vérifier : 91-93 ciblent-ils sm_120 ou sm_100 seulement ?** (79/87 sont GeForce) | NVIDIA/cutlass arbre main |
| Pièges CUTLASS | `-DCUTLASS_NVCC_ARCHS=120a` obligatoire (#2820) ; DSL Python limitait le FP4 à sm_100a (#2800, clos nov. 2025) ; grouped GEMM bloc-échelle = garbage sans `compute_120f` (#3096, mars 2026) | même famille que le piège `-arch=sm_120a` seul de poste4 | cutlass #2820/#2800/#3096 |
| `fp4-cuda-kernel` (VincentKaufmann) : 85-129 TFLOPS sur GB10, 1,4-2,4× bf16, CUTLASS 3.8, poids quantifiés une fois | petit, lisible ; note : **MXFP4 (E8M0) inutilisable sur les tensor cores sm121, seul NVFP4 UE4M3 avec layout `SfKMajorAtom`** | référence de layout d'échelles | GitHub |
| SGLang PR #29190 (juin 2026) : noyau MoE NVFP4 « b12x » pour SM120 : TPOT 18,18 ms contre 19,36 (cutlass) et 22,65 (triton) sur Qwen3.5-397B | +6 % sur le meilleur existant | où en est le noyau MoE natif SM120 chez eux | sgl-project |
| vLLM #31085 (ouvert) : SM120 retombe sur Marlin ; commits `nvfp4_scaled_mm_sm120_kernels.cu`, `nvfp4_blockwise_moe_kernel.cu` (`ENABLE_NVFP4_SM120`) | leurs noyaux SM120 existent dans `csrc/quantization/fp4/` | comparaison noyau à noyau possible | vllm-project |

### 2.4 Attention, KV, MLA, GDN sur SM120

| ressource | fait | pour nous | source |
|---|---|---|---|
| **vLLM PR #50288 (fusionné 29/07/2026) : cache KV NVFP4 sur SM120** via FlashInfer FA2 natif (déquantifie FP4 → BF16 dans le noyau) ; bogue de swizzle des échelles V ; branche `jethac/flashinfer sm120-nvfp4-release-v0.6.15` | −50 % d'octets KV par rapport à int8 | attention = 49,5 % du pas (poste3) → prédiction −10-15 % du pas à ctx ≥ 8k ; **tue** : PPL +>0,5 % ou accord glouton < 98 % (relevés extérieurs : Gemma −0,7 %, DeepSeek +3 % *(résumé)*) | vllm PR #50288 ; llama.cpp #26989 (demande ouverte) |
| FlashInfer #2655 (clos 28/02/2026) : MLA décodage SM120 = **FP8 seulement** (`mla_sm120.cu`), BF16 infaisable dans 99 Ko de shared ; BF16 retombe sur FA2 | aucune référence externe MLA BF16 sur sm120 : notre noyau MLA n'a pas de concurrent à copier | — | flashinfer |
| FlashInfer #5095/#5135 (sept. 2026) : MLA creuse DSv4 corrompue sur SM120 au-delà de 64 jetons | l'écosystème SM120 est encore fragile : nos noyaux maison ne sont pas un détour | — | flashinfer |
| FlashInfer RFC #3628 (juin 2026, ouvert) : **l'instruction bloc-échelle n'est pas bridée sur sm120** (micro-banc 5060 Ti) ; propose une attention prefill MXFP8 en `mma.sync kind::mxf8f6f4` ; SageAttention3 (NVFP4) cité comme référence existante | voie pour une attention prefill en FP8 natif sur notre carte | à suivre | flashinfer |
| Attention FP4 fusionnée maison sur 5070 Ti : 2,4-3,4 TFLOPS contre SDPA 15 — **bornée par la quantification, pas le calcul** | avertissement : FP4 dans l'attention n'est pas un gain gratuit | — | florianmattana.com |
| **FlashQLA-Blackwell** (fork MIT des noyaux TileLang GDN de Qwen, porté sm_120/121) : **2,76×** contre le Triton FLA de vLLM à B=1, T=32 768 (prefill) | Qwen3.6-35B a des couches `linear_attention` : référence pour `docs/CHANTIER-GDN.md` | à comparer à notre `gdn.py` | Plaaasma/FlashQLA-Blackwell |
| llama.cpp PR #26001 : GDN chunké au prefill, +5,4-5,7 % (pp512-8192), réglé sur GA102 | ordre de grandeur modeste | — | ggml-org |
| PersistentKV (2606.26666) : sur RTX 3060, file de travail par groupe de têtes KV réduit les lancements de 16 → 2 par pas, 1,04-1,08× (B8), 1,40× (B1 long contexte) ; une politique calibrée choisit FlashInfer aux petits lots | même question que notre grille `paged_attn_partial` (poste3) | — | arXiv |
| Batch-1 : H100 n'atteint que **27 %** de son plancher mémoire, L4 81 % (2605.30571) ; « la mémoire plus rapide ne se traduit pas en latence » | notre 1 050 / 1 792 = 58 % est dans cette famille ; le terme manquant est la latence/le lancement, pas la bande | — | arXiv |

### 2.5 Graphes, lancements, mégakernels

| ressource | fait | pour nous |
|---|---|---|
| Blink (2604.07609) : lancement de graphe **côté carte ≈ 2 µs** contre 11-17 µs hôte ; −48,6 % d'énergie par jeton (avec SmartNIC) | le lancement hôte est un poste mesurable chez nous (plomberie 2,45 ms) |
| Foundry (2604.06664) : sérialiser les graphes CUDA capturés (adresses + binaires) pour le démarrage à froid | notre `MAX_GRAPHS=16` et la capture par clé : un cache de graphes sur disque est possible |
| MPK (2512.22219, Apache-2.0) 1,2-6,7× ; Ada-MK (2605.11581) : lancements = 14,6 % du temps ; **AutoMegaKernel (2606.09682) : cible sm_120, un mégakernel int8 W8A16 bat cuBLAS bf16 + graphes à b=1** | borne haute pour « ce qui ne devrait pas être dans le pas » ; tue : notre surcoût de lancement < 10 % du pas |
| PDL : llama.cpp l'a refusé (#15479, « beaucoup de changements pour un gain marginal, surtout petits modèles ») ; TRT-LLM l'utilise | priorité basse |
| vLLM : graphes « full » seulement en décodage pur ; « les graphes CUDA sont le facteur n°1 du débit vLLM sur Blackwell consommateur » (elevata) ; Foundry note des captures de plusieurs minutes | nous capturons déjà ; le coût est notre clé 4D (poste3) |

### 2.6 Spéculation (chantier ouvert chez nous)

| fait | chiffre | source |
|---|---|---|
| MTP Qwen3.6 dans llama.cpp | 1,4-2,2× ; 27B sur 3090 : 38 → 65 t/s (1,71×) | mer.vin, jarvislabs *(résumé)* |
| vLLM + MTP n=3 + KV FP8 sur PRO 6000 (même bande que 5090) | eager 24 → graphes 75 → **100-125 t/s** (27B INT4) ; acceptation par position 0,87/0,72/0,60, moyenne 73 % ; n=5 = +4 % seulement ; MoE 35B-A3B FP8 ≈ 170-200 t/s | lastloop-ai/vllm-blackwell-guide (mai 2026) |
| MTP au prefill | **×3,4 plus lourd** ; lookahead plus profond = −24 % ; désactiver pour les longs prompts | note.com |
| n-gram k4v256 sur édition de code | **+93 %** (207 → 490 t/s), sans modèle brouillon | note.com |
| DSpark v2 (gittensor) | chat 1,80×, **code 4,32×** (τ = 6,78), 16k : 1,47× ; « lossless » vérifié par régénération sans brouillon | HF gittensor |
| DFlash (vLLM `--speculative-config method=dflash`, 6 jetons) | 123 t/s à 8k, 90 à 60k, acceptation 59-68 % ; 297 t/s sur 35B-A3B | HF #132, note.com |
| Borne colibrì | −32 % à 85 % de succès | revue colibrì |

Conséquence pour `protocole-ngram-mtp.md` : **n-gram d'abord** (0 poids, +93 % sur le
code, tue : acceptation < 30 %), MTP ensuite (nous avons des variantes `-MTP` dans
le parc), et la mesure publie prefill **et** décodage séparément (×3,4 au prefill).

### 2.7 Deux GPU hétérogènes

Rien de publié pour 5090 + 3080 Ti. Guides génériques : la carte lente borne chaque
synchronisation, PCIe coûte ~25 %, x8 divise la bande. 2603.12707 : couper à la
frontière de modalité minimise le transfert. FreeToken tranche autrement : les ratés
vont **au CPU**, pas à un second GPU. Ma piste « 3080 Ti calculante » (§6 d'hier)
reste sans précédent extérieur ; la mesure (i) de cette note (saut aller-retour
4 Ko) décide, et la piste CPU (§2.2) est son témoin à coût nul.

### 2.8 Sceptiques utiles

* 2606.21428 : sur M2 Pro et Jetson, OLMoE fait **moins bien** qu'un dense de même
  taille active (−10 % / −31 %, 2,1× l'énergie par jeton) ; le routage < 9 % du
  calcul MoE. Un MoE n'est pas économe par construction : à mesurer contre un dense
  apparié (notre témoin MoE bf16, `protocole-temoin-moe.md`).
* Tom's Hardware (Qwen3.8-27B sur 5090) : « la VRAM seule ne compense pas des
  goulots logiciels sévères » *(résumé)*.
* note.com (mai 2026) : « NVFP4 recommandé par NVIDIA, SGLang et vLLM ont tous
  perdu contre llama.cpp en mono-utilisateur » ; MXFP4 = +9 %.

## 3. Ordre recommandé (indépendant de la note d'hier, à fusionner par chef)

| # | geste | coût | décide |
|---|---|---|---|
| 1 | `lspci -vv` : largeur négociée des deux cartes aujourd'hui (M0) | 1 min | 18,7 ou 52,8 |
| 2 | Horloge SM verrouillée en décodage, J/jeton par `energie.py`, jumelles | 1 h carte | −20-30 % attendu |
| 3 | Taxe de stationnement : W au repos avec 0/1/3 contextes | 30 min | ~50 W par contexte ? |
| 4 | M1 (trace) avec le protocole 2608.07911 + contre-trace llama.cpp | 2 h | Δh, et le prior 0,85-0,95 |
| 5 | Vérifier `mlp_exec=cpu` MoE dans le moteur ; micro-banc 8 experts sur CPU | 2 h | CPU contre PCIe |
| 6 | n-gram k4v256 sur nos prompts de code | 2 h | +93 % annoncé |
| 7 | SparkInfer et FreeToken dans `banc-4moteurs` (licences à lire ; FreeToken Apache-2.0) | 1 j | les deux témoins natifs Blackwell |
| 8 | KV NVFP4 dans `paged_attn` (recette vLLM #50288) | 3-5 j | −10-15 % du pas, si PPL tient |
| 9 | GEMM prefill TileLang sm120 (bead brd) | 2-3 j | seulement si le GEMM redevient le goulot |

## 4. Non vérifié, dit tel quel

* Les chiffres marqués *(résumé)* viennent d'un résumé de moteur de recherche,
  pas de la page ; les autres ont été lus dans la page.
* Un relevé HF/blog = un exemplaire, souvent Windows/WSL2 (vllm-blackwell-guide
  mesure +27 % en passant à Linux natif) ; aucun n'a de jumelle.
* Les exemples CUTLASS 91-93 : cible sm_120 non vérifiée.
* SparkInfer : licence et ouverture du code non lues (« SN74 on Gittensor »).
* Aucun chiffre ci-dessus n'a été reproduit sur notre machine.

## 5. Sources

Énergie : arXiv 2605.11999 · 2501.08219 · 2605.23918 · 2601.22076 · 2608.28044 ·
2505.06371 · 2608.00008 · 2606.16106 · 2603.20224.
Cache d'experts : arXiv 2608.18261 · 2608.07911 · 2608.12103 · 2608.16157 (FreeToken,
github.com/FlashML-org/FreeToken, zhongzhuzhou.org/blog/2026-08-23-freetoken-technical-review-en) ·
2605.27081 (ReMoE) · 2502.05370 (FineMoE) · 2509.07379 (DuoServe) · 2512.16473 ·
2605.17889 (CoX-MoE) · 2504.05897 (HybriMoE) · 2511.14102 (MoE-SpeQ) ·
github.com/ggml-org/llama.cpp/discussions/24528 · github.com/ggml-org/llama.cpp/pull/25294 ·
github.com/seppegadeyne/expertpin · github.com/kvcache-ai/ktransformers.
SM120 : github.com/lna-lab/blackwell-geforce-nvfp4-gemm · github.com/tile-ai/tilelang
(`examples/gemm_sm120/`) · research.colfax-intl.com (optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120) ·
github.com/NVIDIA/cutlass/issues/3096, /2820, /2800 · github.com/flashinfer-ai/flashinfer/issues/2723, /2655, /3628, /5095 ·
github.com/VincentKaufmann/fp4-cuda-kernel · github.com/sgl-project/sglang/pull/29190 ·
github.com/vllm-project/vllm/issues/31085, /pull/50288 · github.com/ggml-org/llama.cpp/issues/26989, /15479, /pull/26001 ·
github.com/Plaaasma/FlashQLA-Blackwell · florianmattana.com/posts/fp4-fused-attention-kernel-sm120 ·
forums.developer.nvidia.com (sm121 356 TFLOPS ; qwen3-next 20→35 tps).
Bancs 5090 : zenn.dev/toki_mwc/articles/rtx5090-qwen36-35b-a3b-llmacpp-bench · note.com/unco3/n/n2d702cbf1bb4 ·
note.com/unco3/n/n8212ee12811b · huggingface.co/Qwen/Qwen3.8-27B/discussions/132 ·
huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/discussions/24 · cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000 ·
github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120 · github.com/lastloop-ai/vllm-blackwell-guide ·
github.com/gittensor-ai-lab/sparkinfer · huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 ·
gilesthomas.com/2026/07/benchmarking-qwen-3-6-35b-moe-rtx-3090 · elevata.io/en/nvfp4-inference-blackwell-sm120-gpus-what-worked ·
github.com/ggml-org/llama.cpp/discussions/19890 · github.com/noonghunna/club-3090/issues/1261.
Graphes/mégakernels : arXiv 2604.07609 (Blink) · 2604.06664 (Foundry) · 2512.22219 (MPK) ·
2605.11581 (Ada-MK) · 2606.09682 (AutoMegaKernel) · 2605.30571 · 2606.26666 (PersistentKV).
Sceptiques : arXiv 2606.21428 · tomshardware.com (benchmarking-qwen-3-8-27b-on-rtx-5090-and-beyond).
