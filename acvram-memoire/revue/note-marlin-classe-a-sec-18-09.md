# Note — la classe Marlin (vLLM v0.29.0, `csrc/libtorch_stable/quantization/marlin`, `moe/marlin_moe_wna16`, Apache-2.0) appliquée à notre pile : faisable en 3-5 j **oui, comme PORT** ; porte 110 TFLOPS **corrigée à 137** (poste4, 18/09, à sec, aucune mesure)

1. **Mécanisme** (lu dans `marlin_template.h`, 2 081 l., `dequant.h`, `marlin.cu`) : tuile par bloc de 256 fils = 16·thread_m_blocks (16-64) lignes × thread_n ∈ {64, 128, 256} × thread_k ∈ {64, 128}, configs prioritaires (k,n,fils) = (128,128,256), (64,128,128), (128,64,128) ; **cp.async à 4 étages** global→shared pour A (bf16), B (4 bits) et les échelles ; `mma.sync m16n8k16` bf16, accumulation fp32 ; découpe en « stripes » sur N et K avec réduction entre blocs (stream-K), `max_par` lots de M.
2. **Décodage E2M1 → bf16 en registres, sans table** : masque + décalage (`(q & 0x70007000) >> 6`, signe `0x8000`), deux paires par mot de 32 bits, puis le biais d'exposant (2⁶−2¹, ×2⁻⁷ replié) et l'échelle de bloc en UN `hfma2` — l'échelle E4M3 est convertie au repack en un bf16 « S0E5M3 » (`nvfp4_marlin_process_scales`, facteur 2ⁿ commun aux experts d'une couche), l'échelle globale ×2^(126−7) appliquée en fp32 dans l'épilogue. Les poids sont **pré-permutés au repack** (`gptq_marlin_repack`) dans l'ordre des fragments MMA : aucun `prmt` ni `ldmatrix` de transposition à l'exécution — c'est ce qui manque à B1' (58 TFLOPS : table + `tl.join` + transposition dans la boucle K).
3. **Format d'entrée = le nôtre** : NVFP4 `[N, K/2]` paires E2M1 uint8, `[N, K/16]` E4M3, global fp32 (`NVFP4Tensor`) est exactement ce que `prepare_nvfp4_moe_layer_for_marlin` ingère (group_blocks = 1, `kFE2M1f` + `kFE4M3fn`, K multiple de 64, N arrondi à 64 : 768 et 2 048 conviennent). `marlin_moe_wna16` prend les jetons TRIÉS par expert avec offsets — notre `_forward_prefill_grouped` les a déjà (`tuiles_un_expert`, `_tuiles`). Blackwell GeForce (sm_120) : `mma.sync` Ampère, c'est sur cette carte que vLLM fait ses 20 988 j/s.
4. **Faisable en 3-5 j : oui, en PORTANT** le template (repack au chargement de nos piles — un noyau, à la construction des `_stacks` —, wrapper grouped avec nos offsets, épilogue act·up fusionné, tests ± 2⁻⁷ + bras cassant échelle décalée) ; **non en réécrivant** de zéro (pipeline cp.async + fragments + stripes + repack : 6-10 j et une classe d'erreurs qu'un port n'a pas). Licence Apache-2.0 : réutilisable avec l'en-tête d'attribution, contrairement à TabbyAPI (AGPL, lecture seule). Coût caché : le repack double la VRAM d'une pile pendant sa conversion (768 Ko par expert, faire expert par expert) et fige le format en mémoire (le GEMV de décodage lit `[N, K/2]` tel quel : garder les deux dispositions, +0 octet sur disque, ×2 en VRAM pour les experts — 1,7 Go sur Coder — ou re-permuter à la volée pour le décodage : à trancher, c'est le vrai coût du port).
5. **Porte 110 TFLOPS : incohérente avec le scellé 15 700 j/s, corrigée à 137.** Experts Coder 2 048 : 2·16 384·2 048·768 × 3 projections × 48 couches = **7,43 TFLOP** (B0 : 79,9 ms = 93 TFLOPS, cohérent). Pas actuel 207 ms ; retirer la déquant 50,8 laisse 156 ; 15 700 j/s = 130,4 ms exige experts ≤ 54,3 ms = **137 TFLOPS** effectifs (68 % de la crête bf16 mesurée, 200). À 110 TFLOPS : 67,5 ms → 143,8 ms → 14 240 j/s — la porte serait franchie et le scellé manqué. Un scellé = un seuil : soit porte 137 et scellé 15 700, soit porte 110 et scellé 14 200 (à poste7). Ce que la classe rend sur ces formes (M ≈ 128 lignes par expert, N 768, K 2 048 : intensité 512 FLOP/o, jamais bornée par la bande ; borné par MMA + déquant CUDA-cores recouverte) : vLLM à 20 988 j/s (97,6 ms le prefill entier) implique experts ≤ 60 ms ⇒ **≥ 125-150 TFLOPS** sur cette carte — 137 est atteignable par un port fidèle, pas garanti par un port approximatif (stripes et `max_par` mal réglés : −20 %).
6. Ce que la note ne dit pas : le PPL (mêmes codes W4, A bf16 : ± 0,002 attendu, pas 0), et la place du repack sous graphes (hors décodage : aucun). Décision § 5 de poste7 inchangée : port ou non = semaine CUDA, à l'utilisateur.

## Micro-banc P1 livré à sec (18/09, après la décision de poste7 : porte 137, formule j/s = 2048/(0,076 + 7,43/X))

- `acvram/kernels/marlin_port/` : sources vLLM v0.29.0 verbatim (Apache-2.0,
  `LICENSE-vllm` joint) — `marlin.cuh`, `marlin_dtypes.cuh`, `dequant.h`,
  `marlin_mma.h`, `gptq_marlin_repack.cu`, `moe/marlin_moe_wna16/{kernel.h,
  marlin_template.h, ops.cu}`, `core/scalar_type.hpp`, `torch_utils.h` ;
  espace de noms des ops renommé `acvram_marlin` (`bindings.cpp`, schémas
  copiés) ; instanciations générées pour NVFP4 seul (E2M1 + E4M3 par 16,
  bf16 : 15 noyaux, `sm80_kernel_bfloat16_fe2m1f_bfloat16.cu`). Compilation
  à sec contrôlée : `nvcc -c` des trois .cu + `g++ -c` des bindings sans
  erreur (C++20, `-DUSE_CUDA` pour le shim de l'ABI stable, sm_86 + sm_120).
- `marlin_port/__init__.py` : `charger()` (cpp_extension.load, cache
  `~/.cache/acvram/marlin_port`), `preparer_pile(qw, bs, gs)` = repack par
  expert (`gptq_marlin_repack`, notre `[N, K/2]` uint8 vu int32 et transposé
  comme le nvfp4 de vLLM — même ordre de quartets, bas d'abord) + échelles
  permutées et converties S0E5M3 (facteur 2ⁿ commun aux experts, échelle
  globale × 2^(126−7) ÷ facteur), `aligner_blocs` (moe_align_block_size en
  torch, tri stable), `gemm_moe` (fp32 reduce, sans atomique). L'id
  `kFE2M1f` (562 949 953 487 106) contrôlé contre vLLM.
- `outils/banc-marlin-p1-18-09.py` : formes Coder 2 048 (E 128, top_k 8,
  T 16 384, gate+up N 1 536, down), rejeu de graphe, juge 2⁻⁷ × Σ|x·w| contre
  la déquant fp32 sur 512 lignes, témoin déquant bf16 + cutlass par expert
  (classe B0), X = 7,43 TFLOP / (ms × 48), trois bandes de poste7 imprimées,
  formule Coder et GLM (7,11 TFLOP routés, part fixe donnée ou trois
  hypothèses). JSON `scratchpad/banc-marlin-p1-18-09.json`.
- Tests à sec : `tests/test_marlin_port_a_sec.py` (id, alignement = contrat
  vLLM, échelles). Suite 849 passed.
- Non fait, sur décision : intégration moteur (deux dispositions d'experts,
  Plan, `experts_layout=double` dans `regime_ligne()`), tests ± 2⁻⁷ + bras
  cassant en situ — après le oui de l'utilisateur ; aucune passe de carte
  P1 avant (le banc attend aussi).

Prédiction scellée pour le banc (quand il tournera) : X = 120-150 TFLOPS
(vLLM 20 988 j/s ⇒ ≥ 125 sur cette carte ; port fidèle, block_size_m 64,
thread_k/n choisis par Marlin) ; bande 110-137 plus probable que ≥ 137 (les
tuiles N = 768/1 536 sont petites pour ses stripes) ; témoin B0 ≈ 90-100.
Faux si < 110 (alors la porte se ferme sans carte, comme écrit).

### Conditions de poste7 intégrées au banc (18/09, à sec)

1. JIT hors capture (REGLES § 6, garde b) : `--compiler-seulement` compile
   l'extension SANS carte (`CUDA_VISIBLE_DEVICES=""`, nvcc seul, 23 s ici,
   contrôlé de bout en bout : .so chargé, ops enregistrées) ; le banc charge
   depuis le cache, chauffe en eager (3 appels + 2 sur flux annexe) avant
   toute capture, et se déclare **BANC INVALIDE** si l'extension a été
   compilée dans son processus ou si le sha du .so change pendant le banc.
   Trouvé au passage : depuis CUDA 12.8, nvcc donne une liaison interne aux
   stubs hôte des gabarits `__global__` instanciés explicitement — édition de
   liens « undefined hidden symbol Marlin<…> » ; vLLM pose
   `-static-global-template-stub=false` (CMakeLists.txt:1377), posé aussi.
2. En-tête (INDEX) : `regime_ligne()` + `marlin_port_so=<sha256 du .so>` +
   `marlin_source=vLLM v0.29.0`, imprimés et dans le JSON.

### Prédiction GLM par la formule (écrite avant la passe de poste3, 18/09)

GLM-4.7-Flash : 46 couches MoE, E = 64, top_k 4, H 2 048, I_moe 1 536 →
experts routés **7,11 TFLOP** par préfill de 2 048 (l'expert partagé est un
GEMM dense, hors formule). Dernier préfill mesuré : **4 422 j/s** (W4A16,
verdict-duel-glm-prise-b-16-09, pp2048 ; vLLM 26 732) = 463 ms. Experts
aujourd'hui (B0 ≈ 93 TFLOPS + déquant NVFP4 : 302 Mo/couche, la même
masse que Coder) ≈ 76 + 50 = 126 ms ⇒ **part fixe ≈ 337 ms** (MLA de
préfill par séquence, expert partagé, tête) — c'est elle qui domine, pas les
experts. Formule : j/s = 2048 / (0,337 + 7,11/X) :
X = 110 → **5 100** ; X = 137 → **5 260** ; X = 150 → 5 300 (+15-20 % ; la
parité vLLM ne se joue pas sur les experts). À rebaser sur le témoin T de
la même passe si le pas GLM actuel diffère de 463 ms (B0 est passé défaut
après le 16/09) : fixe = pas_T − 126. Faux si GLM en situ < 0,95 ×
formule(X) — alors la part fixe a bougé ou l'expert partagé pèse plus.
