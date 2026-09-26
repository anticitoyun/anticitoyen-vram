# Veille GitHub — inférence LLM sur RTX 5090 / sm_120 (24 septembre 2026)

Méthode : recherche de dépôts par l'API GitHub (32 requêtes combinant « RTX 5090 », « sm_120 »,
« sm120 », « nvfp4 », « mxfp4 », « fp4 gemv », « gated deltanet », « qwen3.5 / 3.8 inference »,
« cutlass sm120 », « marlin fp4 », « decode kernel », « flashinfer sm120 », « single 5090 »…),
~480 dépôts vus, filtrés sur l'activité 2026 et le contenu réel (noyaux, mesures). La recherche
de *code* GitHub n'a pas pu tourner (API non authentifiée sur cette machine) ; compensée par
WebSearch et la lecture directe de fichiers sources. Digests par gitingest (API `/api/ingest`),
compléments par `raw.githubusercontent.com`. Aucun calcul lancé localement.

Tous les chiffres ci-dessous sont **ceux publiés par les dépôts**, non reproduits ici. Quand un
chiffre est une déduction de ma part, il est marqué *(calcul)*.

---

## 1. Tableau des dépôts retenus

| # | Dépôt | ★ | Dernier push | Ce qu'il apporte en une ligne |
|---|---|--:|---|---|
| 1 | [gittensor-ai-lab/sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer) | 83 | 2026-09-24 | Moteur C++ 5090, Qwen3.8-27B NVFP4 ; GEMV NVFP4 multi-lignes documentée, préfill CUTLASS sm120 avec indices L2, GDN par blocs WY. |
| 2 | [signalnine/q27](https://github.com/signalnine/q27) | 43 | 2026-09-22 | Moteur Qwen3.6/3.8-27B mono-5090 ; décomposition mesurée d'un pas de vérification (1 643 nœuds), réutilisation d'état GDN, coûts hôte. |
| 3 | [kekzl/imp](https://github.com/kekzl/imp) | 41 | 2026-09-24 | Moteur C++23/CUDA sm_120a seul ; FA2 sm120, GEMV NVFP4, recette de quantification Qwen3.8 avec mesures PPL par groupe de couches. |
| 4 | [kacper-daftcode/qwentin](https://github.com/kacper-daftcode/qwentin) (+ fork SRSWTI/knivesysl) | 18 | 2026-08-18 | Qwen3.8-27B FP6 E2M3 en MMA block-scaled, KV 4 bits, 256k ; échelle de qualité FP8/FP6/E2M1/NVFP4. |
| 5 | [Neroued/ninfer](https://github.com/Neroued/ninfer) via [splickz/ninfer-rk4v4-e8](https://github.com/splickz/ninfer-rk4v4-e8) | 2 (fork) | 2026-08-31 | Concurrent de référence : allocation mixte FP8/NVFP4 de Qwen3.8, W4A4 au préfill / A16 au décodage sur le même artefact. |
| 6 | [shiinamiyuki/sm120_gemm](https://github.com/shiinamiyuki/sm120_gemm) | 17 | 2026-08-31 | GEMM sm120 sans dépendance : W4A16 petit-M sur cœurs tenseurs (déquant en registres), W4A4 natif, analyse du coût fixe. |
| 7 | [net-snix/vllm-hybrid-nvfp4](https://github.com/net-snix/vllm-hybrid-nvfp4) | 12 | 2026-07-05 | Aiguillage par M : Marlin W4A16 sous 128 jetons, CUTLASS W4A4 au-dessus ; +95 % de préfill. |
| 8 | [tie-pilot-qxw/pdl-megakernel-reconstruction](https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction) | 22 | 2026-08-05 | 81 noyaux chaînés par PDL = 97,4 % d'un mégakernel ; où placer le déclencheur. |
| 9 | [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) | 63 | 2026-02-25 | Mégakernel décodage 5090 ; débit lecture mesuré, préchargement L2 pendant l'attention, barrières. |
| 10 | [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) | 540 | 2026-09-23 | Noyaux SASS sm120 écrits à la main, GEMV MXFP8 et MoE FP8 ; pièges de graphe CUDA sur sm120. |
| 11 | [xinyang-zhou/sm120_nvfp4_ops](https://github.com/xinyang-zhou/sm120_nvfp4_ops) | 4 | 2026-09-24 | GEMM NVFP4 CuTe sm120 (split-K spécialisés M=128/256), attention et MoE. |
| 12 | [lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm) | 25 | 2026-04-27 | Carte des manques sm120 (pas de tcgen05/TMEM, 99 Ko smem) et 12 correctifs vLLM/FlashInfer/CUTLASS. |
| 13 | [Plaaasma/FlashQLA-Blackwell](https://github.com/Plaaasma/FlashQLA-Blackwell) | 18 | 2026-06-05 | Noyau GDN par blocs (TileLang, Qwen) porté sm12x ; 2,76× vs FLA Triton. |
| 14 | [RightNow-AI/qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | 123 | 2026-02-28 | Qwen3.5-27B en Triton ; profil par catégorie d'un pas de décodage hybride GDN, GDN fusionné. |
| 15 | [blake-snc/sm121-kernels](https://github.com/blake-snc/sm121-kernels) | 5 | 2026-08-14 | 259 noyaux PTX pour sm121 (frère de sm120) : GDN, GEMV W4A16/NVFP4, split-K déterministe. |
| 16 | [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) | 1 681 | 2026-09-23 | Qwen3.8-27B vLLM (3090/5090) : état GDN fp16, activations int8 — mesures de débit et de PPL. |
| 17 | [Nekofish-L/mxfp6_sm120](https://github.com/Nekofish-L/mxfp6_sm120) | 14 | 2026-09-20 | GEMM MXFP6 sm120 ; qualité FP8/MXFP6/NVFP4 mesurée sur Qwen3.8-27B. |
| 18 | [mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix) | 203 | 2026-04-21 | « 4/6 » : choix adaptatif d'échelle NVFP4 par bloc (erreur quadratique), noyaux sm120. |
| 19 | [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) | 2 876 | 2026-09-24 | Premier mégakernel hybride DeltaNet/attention ; brouillons DFlash2 pour Qwen3.8 et gemma4-31B. |
| 20 | [Maharajahu/Qwen3.8-27B-RTX5090-CUDA-Kernels](https://github.com/Maharajahu/Qwen3.8-27B-RTX5090-CUDA-Kernels) | 1 | 2026-09-05 | Attention de vérification MTP sans matérialiser le KV quantifié en FP16 : +49,8 % à 190k. |
| 21 | [BlinkDL/Albatross](https://github.com/BlinkDL/Albatross) | 131 | 2026-09-10 | RWKV-7 7,2B sur 5090 : coût d'un pas b=8 vs b=1 pour un modèle récurrent. |
| 22 | [notwitcheer/sm120-field-guide](https://github.com/notwitcheer/sm120-field-guide) | 25 | 2026-09-09 | 99 pièges sm_120 mesurés (service, quantification, mesure). |
| 23 | [Harry-Chen/fp4_sm120](https://github.com/Harry-Chen/fp4_sm120) | 24 | 2026-07-20 | Instructions FP4 absentes de sm120 (`cvt.rs…e2m1x4`) et contournements bit-exacts. |
| 24 | [fms-zth/BlackFlash](https://github.com/fms-zth/BlackFlash) | 23 | 2026-05-25 | FA2 TMA + spécialisation producteur/consommateur sur sm120 — utile surtout comme contre-exemple. |

Écartés après lecture : listes de recettes Docker (MiaAI-Lab, AEON-7…, pas de noyau), wiki
« blackwell-gpu-wiki » (erreurs factuelles : wgmma et L2 16 Mo annoncés pour sm120), tokenspeed
(orienté GB200/B200), qwen35-thor (Jetson Thor), forks vides.

---

## 2. Repères matériels sm_120 mesurés par ces dépôts

| Grandeur | Valeur | Source |
|---|---|---|
| Lecture DRAM atteignable | 1 674 Go/s (93 % de 1 792) | AlpinDale, billet « 5090 decode optimization » |
| Plafond lecture mesuré | 1 707 Go/s ; GEMV W4A16 en régime ~1 500–1 550 Go/s | sm120_gemm README, § W4A16 |
| MMA FP4 block-scaled | 780–868 TFLOPS (`tools/microbench_mxf4`) ; n'existe que sous `sm_120a` | q27 README, « fp4 note » |
| `mma.sync` FP4 | ½ du chiffre commercial ; accumulation f32 = ¼ du débit | imp `docs/internals/KERNELS.md` § 5 |
| W4A4 cuDNN 4096³ | 1 289 TFLOPS (~77 % du nominal) | sm120_gemm § W4A4 |
| Smem | 99 Ko par bloc (101 376 octets opt-in) ; pas de TMEM, pas de tcgen05, pas de multicast | lna-lab, imp |
| Charges vectorielles | plafonnées à 128 bits (pas de 256 bits sur sm_120) | AlpinDale, § « things that didn't work » |
| L2 | 96 Mo | AlpinDale |
| `grid.sync()` coopératif | ~3 µs par barrière | AlpinDale |

Conséquence rappelée par q27 : NVFP4 transporte 1,06× les octets d'un Q4 groupe-64 à poids égal ;
au décodage c'est le nombre d'octets qui compte, pas les FLOPS.

---

## 3. Chiffres de référence Qwen3.8-27B sur une 5090

| Moteur | Format | b=1 sans spéculation | Autre | Source |
|---|---|---|---|---|
| sparkinfer | NVFP4 uniforme, 17,9 Go | **95,7 t/s** (ctx 128) ; 93,6 à 4k | préfill 14 364 t/s à 4k | README |
| sparkinfer | checkpoint unsloth (NVFP4 MLP + FP8 attention) | 84,9 t/s | préfill 5 031 t/s | README |
| imp | NVFP4 (18,3 Gio avec table d'embeddings) | ~102 t/s | — | README, PROV 2026-08-31 |
| NInfer | nvfp4 mixte FP8/NVFP4 (20 Gio) | 71,2 t/s (7 680 jetons) | préfill 8 340 t/s ; C=8 MTP3 766,6 t/s agrégés | `docs/performance/qwen3.8-27b.md` |
| NInfer | groupwise-int | 79,8 t/s | préfill 3 275 t/s | idem |
| vLLM | NVFP4, sans spéculation | 67,8 t/s | C=8 438,7 t/s | q27, tableau 2026-08-17 |
| qwentin | FP6 E2M3 + KV Q4, MTP profondeur 6 | 21,9 ms/tour | 140,9 t/s froid, 217 t/s suite | README |
| q27 | Q4 G64 + DFlash2 K=7 | tour 17,8 ms = brouillon 2,5 + vérif 15,1 + hôte 0,2 | 228–232 t/s agrégés en trafic agentique | README « State of the engine » |

*(calcul)* sparkinfer à 95,7 t/s sur 17,9 Go = ~1,71 To/s effectifs : le décodage b=1 NVFP4 est
au mur de lecture. L'écart restant chez acvram ne peut donc venir que de ce qui n'est pas lecture
de poids.

---

## 4. Faits utiles par dépôt

### 4.1 sparkinfer — `gittensor-ai-lab/sparkinfer`

**GEMV NVFP4** — `kernels/csrc/cuda/gemm/gemv.cu`
- Décodage e2m1 **arithmétique**, pas par table : `const float t[16]` indexé par un quartet
  dynamique finit en mémoire *locale* ; c'est ce qui bridait `gemv_nvfp4_sk_kernel` à 21–46 % de
  la bande passante quand les GEMV Q4_K dp4a voisines tenaient 86–89 %. Forme retenue : magnitude
  doublée `e ? ((2+m) << (e-1)) : m`, un seul int→float, bit-identique (commentaire l. 782-800).
- **Multi-lignes** : un noyau « rows » qui hisse la lecture DRAM mais re-décode les 16 poids
  pour chaque ligne coûte **1,82×** un appel mono-ligne pour **deux** lignes (au lieu de ~1,05×
  attendu) : passé une ligne, le re-décodage rend le noyau lié au calcul. Correctif : décoder un
  groupe une fois, puis produit scalaire par ligne (l. 885-895).
- Mesures négatives consignées : placer les activations en smem par tuile de 1024 = **−30 %**
  (deux `__syncthreads()` sérialisent les chaînes split-K) ; ncu : 3,41 Go de trafic L1 contre
  29,5 Mo DRAM, pipe mémoire 88 %, SM 33 % — lié à la latence d'une chaîne dépendante, pas à L1.
- Hisser la multiplication par l'échelle hors du groupe : +22 % de vitesse **mais** top-1 tombé à
  0,9849 → refusé (non équivalent).
- **PR #1081** (1,53× à 16 flux) : les projections FP8 du checkpoint unsloth (attention q/k/v/o,
  GDN `in_proj_qkv`/`in_proj_z`/`out_proj`, lm_head, MLP 56-63) retombaient dans une **boucle
  ligne par ligne** au-delà de 8 lignes → chaque poids relu une fois par ligne = **59 % du temps
  GPU à c16**. Passage par paquets de 8 lignes : 316,8 → 485,5 t/s agrégés.
- **PR #1132** : seuil `gu_gemm_min_rows` abaissé de 4 à 2 → la GEMM block-scaled prend le relais
  de la GEMV dès 2 lignes (1,11× à c2).

**Préfill NVFP4** — `kernels/csrc/cuda/fused/prefill_nvfp4_sm120.cu`
- Piège d'inclusion CUTLASS : `arch/config.h` doit précéder `float_subbyte.h`, sinon
  `CUDA_PTX_FP4FP6_CVT_ENABLED` reste indéfini et **tout encodage FP4 compile en logiciel** sans
  avertissement (l. 6-13).
- Politique L2 : à M=128 chaque CTA relit toute l'activation A (down : 150 Mo de trafic L2 contre
  74,8 Mo de poids depuis la DRAM). TMA ignore `__ldcs` et `cudaAccessPolicyWindow` ; seul
  l'opérande `.L2::cache_hint` agit. Poids B/SFB en `EVICT_FIRST` ; mettre A en `EVICT_FIRST`
  mesure **61 % pire** (contrôle falsifiable), A en `EVICT_LAST` = nul (l. 36-59).
- Tuiles longues séquences (m=16384) : 256×128×128 = +6,7 % ; 128×256×128 = −14 % ;
  256×256×64 = −76 % ; les tuiles plus grandes ne tiennent pas deux étages en 99 Ko (l. 171-182).
- LM head en sortie float : arrondir les logits en bf16 crée des égalités dans l'argmax.

**GDN préfill** — `kernels/csrc/cuda/fused/prefill_gdn_chunk.cu`
- Balayage naïf (un warp par colonne d'état, une mise à jour de rang 1 par jeton) = 59,3 ms,
  **20,7 %** d'un préfill de 286 ms à 4k, ~5 % du pic fp32 ; ~200× d'amplification de lecture L2
  (~101 Go de trafic L2 par couche). Remplacé par la forme par blocs WY/UT.
- Numérique : `ssm_a` < 0 partout (min −77) → Γ sous-déborde dans un bloc ; porter G = log Γ en
  espace log, n'utiliser que des rapports exp(G_t − G_s) ≤ 1.
- État GDN compacté fp32 → bf16 en décodage continu (PR mentionnée par la recherche web).

**Qualité publiée** : top-1 0,953, KL 0,031 contre la référence llama.cpp ; portes de précision
à chaque PR (`bench/quality/README.md`).

### 4.2 q27 — `signalnine/q27`

- **Décomposition d'un tour de vérification** (Nsight, état de 28 701 jetons, largeur K=7) —
  `docs/perf-next-2026-09-12.md` l. 157-177 :

  | Poste | ms |
  |---|--:|
  | graphe de vérification, tous noyaux | 14,71 |
  | projections Q4 | 9,09 |
  | attention | 1,13 |
  | RMSNorm/quantification | 0,83 |
  | récurrence GDN | 0,78 |
  | échantillonneur multi-voies | 0,24 |
  | hôte | 0,22 |

  **1 643 nœuds** de graphe par tour. *(calcul)* hors projections : ~5,6 ms pour ~8 positions —
  à comparer aux 13 ms d'acvram à b=8.
- Coût hôte côté service : rendu + encodage du gabarit 38–39 ms → 12 ms (cache BPE borné, ne pas
  encoder deux fois le bloc système) ; TTFT tiède −19,6 ms (−6,4 %).
- État GDN comme objet de premier rang : instantané des 48 états à la dernière frontière stable,
  anneau de points de contrôle en mémoire hôte épinglée tous les 4 096 jetons, cache de préfixe
  persistant vérifié jeton par jeton. Sans cela : 0 % de réutilisation et 2 à 7× de temps mur sur
  trafic agentique (mesuré chez NInfer avant son correctif).
- « Record-then-fold » réduit le coût GDN par emplacement : 8 emplacements là où llama.cpp sature à
  6 (~1 Gio par point de contrôle).
- Batching continu : une seule passe de poids pour tous les emplacements, rounds rejoués en graphes
  CUDA indexés par forme ; 2 emplacements = 1,41×, coût solo ≤ 0,07 %. Plafond du cache de graphes
  à revoir (44+ clés en trafic réel contre 28 en banc).
- Format `docs/FORMAT.md` : Q4 symétrique groupe 64, échelles fp16 à part, un warp lit 128 o =
  256 poids = 4 groupes (lecture coalescée).
- W4A8 préfill avec TMA : 1,33–1,39× seulement → abandonné (`tools/gemm_w4a8_spike.cu`).
- Méthode : « PPL classe les formats de KV à l'envers » (l'erreur signée se compense) — juger sur
  les positions catastrophiques en queue de distribution.

### 4.3 imp — `kekzl/imp`

- Débits (Qwen3-8B NVFP4, serveur) : TPOT p50 **3,4 ms** à 1 flux, 5,1 ms à 8, 5,8 ms à 32 ;
  attente en file p50 2,5 ms (`docs/PERF.md`).
- Qwen3.8-27B NVFP4 : ~102 t/s b=1.
- Attention FA2 sm120 (`src/compute/attention_fmha_sm120.cu`, doc `docs/internals/KERNELS.md`) :
  - QKᵀ et PV en `mma.sync.m16n8k16.f16…f16` (accumulation f16) : le QK f32 tournait au ¼ du
    débit ; gain +3–4 % préfill, **+0,37 % PPL** ;
  - O dans les registres, Q en registres, Bq=128, Bkv=64, 8 warps, anneau `cp.async` 3 étages ;
    softmax via `ex2.approx` sur le pipe SFU en parallèle du pipe tenseur ;
  - repli : Bq=64 + 2 CTA/SM quand la grille sous-remplit les 170 SM ;
  - réfuté : QK en FP4 dans l'attention (PPL 5 722 contre 6,12), anneau plus profond (+9 %/+15 %),
    spécialisation producteur/consommateur.
- GEMM NVFP4 autonome `tools/standalone/gemm_nvfp4_sm120a.cu` : 807/972 TFLOPS (4k/8k) ; le
  diagnostic « lié à L2 » était un débit de *requêtes* (82 % des requêtes, 44 % des secteurs) :
  empaquetage tuile-CTA-majeur (lignes L2 pleines de 128 o) et entrelacement de colonnes
  (`uint2`), mio_throttle 5,77 → 2,74.
- Petit-M / split-K NVFP4 groupé : `src/compute/gemm_grouped_nvfp4_smallM.cu`.
- **Quantification Qwen3.8-27B** (`docs/quantization.md`, `docs/archive/quantization_awq_findings.md`) :
  - restent en précision source : `embed_tokens` (2,4 Gio ; forcer NVFP4 = +0,94 % PPL sur
    0,6B), **tête MTP** (quantifiée : acceptation 81 % → 0) ;
  - lm_head re-quantifié NVFP4 au chargement : PPL 4,5707 → 4,6158 (**+0,99 %**) pour
    78,6 → 86,7 t/s (**+10,4 %**), la tête pesant ~11 % du pas b=1 ;
  - options `--keep-gdn-proj [in,gate,out]` et `--keep-attn-gate` ; garder le q_proj fusionné
    Q+gate en source a mesuré **1,5 % pire** en PPL (contre-intuitif, cause : décalage RMSNorm) ;
  - AWQ : sur GQA large (n_rep ≥ 5) ne calibrer que gate/up et down (`--calib-groups BD`), le
    repliement sur o_proj/v_proj fausse la recherche (+1,90 de PPL d'interaction à n_rep=5) ;
  - piège RMSNorm à décalage unitaire `(1+g)` des Qwen3.5/3.8 : replier une échelle AWQ dans le
    gain stocké en bf16 perd 25 % à s=1,25 et 100 % à s ≥ 2 ; une norme finale sans le `+1` a été
    la cause racine d'un bug de tête (#1287) ;
  - calibration non déterministe → PPL dispersée de ~1,6 % entre deux exécutions : forcer le
    GEMM déterministe pour calibrer.

### 4.4 qwentin (et knivesysl)

- Poids FP6 E2M3 block-scaled (échelles par 128), MMA `mma.sync…kind::mxf8f6f4.block_scale`,
  dépaquetage `ldmatrix…b6x16`, tout en PTX inline compilé par nvcc/ptxas stock ; un seul fichier
  `src__forward_qwen.cu`.
- **Échelle de qualité** (top-1 en teacher forcing contre bf16, même protocole) :
  FP8 95,94 > FP6 91,30 > E2M1 86,46 ≈ NVFP4 85,78.
- Décodage : 21,9 ms/tour à court contexte, 25,1 à 128k, 26–28 à 243k ; préfill froid 2 534 t/s
  (4k) → 826 t/s (243k).
- Préfill « large » : projections en une GEMM FP6 large, GDN par blocs, attention MMA contre le
  KV Q4 ; 2 à 4× plus rapide que le chemin par défaut.
- Attention décodage longue : un CTA de 512 threads, 8 warps lisent le K d'un groupe KV **une
  fois** par super-tuile pendant que 8 autres font P·V de la tuile précédente (smem double).
- État DeltaNet ~155 Mio par client (plancher de concurrence).
- knivesysl (dérivé) : poids « empaquetés directement dans la disposition de fragment consommée
  par les noyaux tenseurs », RMS et SiLU fusionnés dans la quantification NVFP4, GDN WY/UT par
  blocs de 64 avec balayage fp32/tf32.

### 4.5 NInfer (concurrent) — lu via `splickz/ninfer-rk4v4-e8`

- `README-upstream.md` l. 20-27 : le profil **Qwen3.8 nvfp4 est mixte** — NVFP4 uniquement pour
  les MLP des couches 0-55 ; **FP8 par ligne** pour l'embedding, les projections d'entrée/sortie
  de l'attention, les projections GDN Q/K/V/Z et de sortie, la tête de sortie et les MLP 56-63
  (repris d'`unsloth/Qwen3.8-27B-NVFP4`). Le profil Qwen3.6 nvfp4 fait **W4A4 MMA au préfill et
  A16 au décodage** sur le même artefact.
- Scores publiés : GPQA-Diamond 90,40 % en nvfp4 mixte contre 87,37 % en groupwise-int.
- Acceptation MTP3 de Qwen3.8 nvfp4 : 45,8–48,9 % contre 67–71 % pour les autres profils.
- Évaluateur PPL `ninfer-perplexity` : corpus fixe de 16 flux (anglais, chinois, code C++/CUDA),
  fenêtre 4 096, pas 2 048 (`docs/perplexity.md`).
- Fork rk4v4-e8 : KV 4 bits sur réseau E8, 17 980 o/jeton contre 35 220 en int8, décodage à
  parité (60,0 contre 59,7 t/s), préfill −20 %.

### 4.6 sm120_gemm — `shiinamiyuki/sm120_gemm`

- **W4A16 petit M** (`w4a16_gemm_tinym_tc.cuh`) : deux familles choisies par autotune —
  cœurs CUDA (coût ∝ M : 0,0225 → 0,0359 ms de M=1 à 8) et cœurs tenseurs `mma.m16n8k16` à coût
  **plat** (0,0261 → 0,0262 ms de M=1 à 8) ; croisement à **M ≈ 5–6**. La famille tenseur
  déquantifie W **dans les fragments B en registres**, sans tampon bf16 : le fragment que
  `ldmatrix` produirait est, par voie, exactement deux octets d'e2m1 dans un seul bloc d'échelle,
  et le swizzle TMA 128 o rend ces lectures sans conflit.
- Mesures (graphe CUDA, meilleur de) : M=8 N=28672 K=4096 : 0,0446 ms, 1 492 Go/s ; plus rapide
  que FlashInfer `mm_fp4` sur les 6 formes.
- **Coût fixe** : à M=8, 79 % du pic ; en allongeant K le régime atteint 91 % de 1 707 Go/s.
  ~13 % partent en remplissage/vidage TMA et en **lancement séparé de la réduction split-K** ;
  levier restant : replier la réduction dans la GEMM.
- W4A4 natif `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3` : pas de
  déquantification dans la boucle ; dispositions de fragments sondées sur le matériel
  (`bench_w4a4 --probe-mma`, `src/nvfp4_mma.cuh`).
- Dent de scie des vagues : 1 792² (196 tuiles, 2 tours) est 26 % plus lent que 1 536² (144
  tuiles, 1 tour) ; remède : ordonnancement stream-K sur exactement 170 CTA.
- Méthodologie : la même config donne 419/824/940 TFLOPS en moyenne espacée et 528/902/1 199 en
  graphe dos à dos — comparer à l'identique.

### 4.7 vllm-hybrid-nvfp4 — `net-snix/vllm-hybrid-nvfp4`

Formes MLP Qwen3.6-27B (gate_up 34816×5120, down 5120×17408) :

| M | Marlin | CUTLASS W4A4 |
|--:|--:|--:|
| 1 | **0,025 ms** | 0,101 ms |
| 64 | 0,064 ms | **0,045 ms** |
| 16 384 | 19,6 ms (299 TF) | **5,5 ms (1 058 TF)** |

Aiguillage par nombre de jetons (seuil 128, `HYBRID_NVFP4_M_THRESHOLD`) : préfill 32k
4 987 → 9 704 t/s (+95 %), décodage inchangé. Coût : **deux copies des poids** (~+1× la VRAM
FP4) — le vrai sujet pour acvram est d'éviter cette double copie (voir leviers).

### 4.8 PDL — `tie-pilot-qxw/pdl-megakernel-reconstruction`

- H100, Llama-3.2-1B, déjà sous graphe CUDA : 81 noyaux séparés = 967 µs contre 777 µs pour le
  mégakernel persistant (trou de **190 µs**, soit ~2,3 µs par frontière *(calcul)*) ; avec PDL :
  810 µs (trou 32,5 µs).
- Placement (`docs/reconstructing-a-megakernel-with-pdl.md`, § « Two-Line Trick ») : le
  producteur appelle `cudaTriggerProgrammaticLaunchCompletion()` dès qu'il a **émis** ses charges
  de poids (pas quand elles sont arrivées : ce seul déplacement vaut 34 µs) ; le consommateur émet
  ses propres charges de poids/KV puis n'appelle `cudaGridDependencySynchronize()` qu'à la
  première lecture de l'activation.
- Échecs consignés : co-résidence de deux grilles, réduction de noyau pour tenir à deux,
  dépendances par tuile.

### 4.9 qwen_megakernel — `AlpinDale/qwen_megakernel` (+ billet de blog)

- Qwen3-0.6B bf16 : 1 036 t/s (0,97 ms/jeton) ; lecture 1,19 Go par pas.
- Barrières : `grid.sync()` coopératif ~3 µs ; remplacé par une barrière atomique à compteur de
  génération monotone par bloc (évite l'ABA), `fence.acq_rel.gpu` plutôt que `__threadfence()`.
- **Préchargement L2 dans la bande passante libre** : pendant l'attention (16 blocs actifs), les
  112 autres émettent `prefetch.global.L2` sur les poids o_proj + MLP (~23 Mo) ; précharger dans
  une phase déjà saturée coûte 8,7 %.
- RMSNorm redondant dans chaque bloc plutôt qu'une barrière ; charges 128 bits contournant L1.
- Tête LM : argmax en deux phases fusionné en un noyau avec compteur global : −2 à 3 µs/pas ;
  ~1 500 Go/s sur la tête.
- Arguments immuables pour la capture en graphe : position et jeton copiés par `cudaMemcpyAsync`
  depuis un tampon hôte épinglé, remise à zéro des barrières sur le GPU (aucun `cudaMemsetAsync`).

### 4.10 vLLM-Moet — `kacper-daftcode/vLLM-Moet`

- GEMV MXFP8 `tools/dsv41_sm120/sm120_gemv/mxfp8_gemv_sm120.cu` : un bloc possède 8 colonnes de
  sortie, ses 8 warps se partagent les blocs K, **chaque voie émet toutes ses charges de 16 o
  avant de convertir** ; remplace CUTLASS (tuile 128 lignes) pour M ≤ 16.
- MoE FP8 `tools/qwen38_sm120/moe_gemv/fused_moe_gemv_sm120.cu` : activation silu·up et
  quantification UE8M0 de l'entrée du down fusionnées dans le noyau ; bit-identique à Triton ;
  −1,0 ms/pas.
- Graphe CUDA : une table déchargée sur CPU bloquait chaque graphe de décodage ~1,2 ms dans un
  nœud `cuStreamWaitValue32` ; une GEMM « skinny » bf16 CuTe-DSL réservée à B300 fonctionnait sur
  sm120 une fois ouverte (−1,3 ms). Au total 65 → 98,5 pas/s.
- Outillage : bancs L2 froids qui font tourner > 128 Mo de poids (`tools/sm120_perf/`), anatomie
  par graphe.
- Toolchain SASS sm120 (`blackwell-isa`, `cubit`) : `QMMA.SF`, entrelacement du décodage PRMT-LUT
  dans le flux QMMA, registres 64 → 4 CTA/SM.

### 4.11 sm120_nvfp4_ops — `xinyang-zhou/sm120_nvfp4_ops`

- GEMM NVFP4 CuTe (384 threads, TMA 3 étages, CTA persistants) : M=16 N=4096 K=8192 = 25,2 µs
  (104 % de cuBLASLt) ; M=512 : 28,4 µs, 1 208 TFLOPS.
- Chemins spécialisés M=128 split-K=4 et M=256 split-K=2 (réduction `half2`) :
  `src/gemm/specialized/`, aiguillage `src/gemm/gemm_dispatch.cu`.

### 4.12 lna-lab — `lna-lab/blackwell-geforce-nvfp4-gemm`

- sm120 = `mma.sync` de sm80 + TMA de sm90 + block-scale de sm100, **sans** TMEM, tcgen05,
  multicast de cluster ; tuiles limitées à 99 Ko.
- Correctifs listés (patches/…) dont `12-marlin-w4a8-sm120.md` (Marlin FP8 sur sm120) et
  `09-fp4-quantization-sm120.md` (quantification d'activation FP4 JIT).
- Denses NVFP4 sur RTX PRO 6000 via vLLM : Qwen3.5-27B 57 t/s, **Gemma4-31B 51 t/s** (point de
  comparaison bas pour gemma4).

### 4.13 FlashQLA-Blackwell — `Plaaasma/FlashQLA-Blackwell`

- GDN par blocs (TileLang) : B=1, T=32 768, Hk=Hv=64 : 48,9 ms (FLA Triton) → 17,7 ms (2,76×).
- Mais seulement **~3 % de TTFT** sur Qwen3.6-27B à 8k : le préfill y est dominé par les 16
  couches d'attention et les 64 MLP.
- Pièges : `T.gemm_v1(transpose_B=True)` donne des résultats faux sur Blackwell → `gemm_v2`
  (`docs/BLACKWELL_FIXES.md`) ; état attendu en (B, H, K, V) alors que vLLM stocke (B, H, V, K).

### 4.14 qwen3.5-triton — `RightNow-AI/qwen3.5-triton`

Profil d'un pas b=1 bf16 sur B200 (10,8 ms, plancher théorique 6,7 ms) : cuBLAS 8,59 ms (80 %),
**DeltaNet fusionné 0,56**, résiduel+RMSNorm 0,42, conv1d causale 0,37, lancements 0,27,
attention 0,19, SiLU 0,11, KV 0,10, RMSNorm à porte 0,09 → ~2,2 ms hors GEMV.
Noyau DeltaNet fusionné (`forge/kernels/triton_deltanet_fused.py`) : split QKV, L2-norm, porte,
beta, décroissance, règle delta, mise à jour, sortie en un seul lancement par tête ; tête LM +
argmax sans matérialiser les 248k logits (`triton_lm_head_topk.py`). Batch : b=8 = 15,9 ms/jeton
(501,9 t/s agrégés).

### 4.15 sm121-kernels — `blake-snc/sm121-kernels`

PTX écrit à la main pour sm121 (même ISA MMA que sm120) : GDN (préfill par blocs, décodage MMA,
décodage TMA, mise à jour d'état, conv1d+SiLU), GEMM W4A16 / NVFP4 / MXFP4, split-K avec variante
**déterministe par construction** (`SPARK_DETERMINISTIC=1`). Inventaire :
`docs/kernel_inventory.md`, guide `docs/sm120_architecture_guide.md`.

### 4.16 HyperQwen — `syv-ai/HyperQwen`

Qwen3.8-27B vLLM, 64 flux (batch/README.md) :

| Config | décodage stable (t/s) |
|---|--:|
| W4A16, état GDN fp32 (37 requêtes sur 64 tiennent) | 516 |
| W4A16, **état GDN fp16** | 707 |
| + activations int8 sur gate/up | 787 |
| + int8 sur tout le MLP (défaut) | 876 |
| + int8 sur toutes les linéaires | 1 025 (PPL +3,7 %) |

Le passage de l'état GDN en fp16 vaut **+37 %** à forte concurrence.

### 4.17 mxfp6_sm120 — `Nekofish-L/mxfp6_sm120`

Qwen3.8-27B, 256 enregistrements en teacher forcing, MAE moyenne des logprobs contre bf16 :
**FP8 0,057 ; MXFP6 0,091 ; NVFP4 0,177**. GEMM MXFP6 ~2× plus rapide que FP8 à M=1..8.
Producteurs fusionnés : « GDN gated-RMSNorm / préparation MXFP8 alimentant la projection W6A8
en PDL » (`gemm_from_gdn`, `gemm_from_swiglu`).

### 4.18 fouroversix — `mit-han-lab/fouroversix`

« 4/6 » : pour chaque bloc NVFP4, choisir entre échelle de max 6 (standard) et max 4 selon
l'erreur quadratique (`scale_rule`), même format sur disque, compatible avec les noyaux
existants ; noyaux compilables pour sm120 (`CUDA_ARCHS=120`). Article arXiv 2512.02010.

### 4.19 lucebox — `Luce-Org/lucebox`

- Mégakernel hybride DeltaNet/attention (Qwen3.5-0.8B) : récurrence DeltaNet en registres F32
  par warps coopératifs ; `grid.sync()` à l'intérieur de la boucle de récurrence = blocage
  silencieux → synchroniser entre couches seulement ; variante NVFP4 `kernel_gb10_nvfp4.cu`.
- Brouillons DFlash2 : Qwen3.8-27B (6,4× sur R9700 contre AR), **gemma4-31B 3,2×** (Q8_0).
- 5090 : Qwen3.8-27B 110,6 t/s sur prompt de 26 758 jetons.

### 4.20 Maharajahu — attention de vérification MTP

Ne plus matérialiser tout le KV quantifié en FP16 à chaque vérification MTP : décodage Qwen3.8
à 190k 53,8 → 80,5 t/s (+49,8 %), sorties bit-identiques (`docs/kernel-design.md`,
`patches/validated-v5.patch`).

### 4.21 Albatross — `BlinkDL/Albatross`

RWKV-7 7,2B fp16 sur 5090 : b=1 6,86 ms/pas, **b=8 8,53 ms** (+24 %), b=32 11,1 ms. Montre ce
qu'un modèle récurrent peut coûter en plus par ligne quand l'état est bien géré (FFN creux sans
perte, disposition d'état [K,V], « DeltaLog »).

### 4.22 Divers

- sm120-field-guide : KV q4_0 = −17 à −23 % de décodage à 16k sur 27B/31B ; la capture de graphe
  alloue au-delà de la fraction mémoire réservée ; MXFP4 QuTLASS « gagne 4× à la GEMM, perd 3-4×
  au jeton ».
- fp4_sm120 : `cvt.rs.satfinite.e2m1x4.f32` (arrondi stochastique) absent de sm120 ;
  `cvt.rn.satfinite.e2m1x2.f32` présent (`stochastic_rounding/sr.sm120.cu`).
- BlackFlash : FA2 TMA + producteur/consommateur à 62–73 % de cuDNN seulement — confirme la
  conclusion d'imp (sur sm120, `mma.sync` bloque le warp émetteur ; la spécialisation ne paie pas).
- vLLM : la voie GDN fusionnée MTP exige `num_v_heads == 8 * num_k_heads` ; Qwen3.8-27B a 48/16 = 3
  → repli silencieux à chaque pas (discussion Hugging Face Qwen/Qwen3.8-27B #51 : MTP devient une
  perte nette de 3,6×). À vérifier si acvram a une garde de forme équivalente.
- arXiv 2607.16831 : décodage GDN sur B200 à ~9,3 µs par appel (concours FlashInfer MLSys 2026).

---

## 5. Leviers pour acvram, classés par gain probable

Rappel des manques : pas b=8 = 13 ms hors lecture des poids (NInfer : 16 ms de pas total) ;
4,8 ms perdues côté service/hôte ; dépaquetage Marlin au préfill (+40–50 ms/requête) ; GDN ;
PPL nvfp4 +11 % contre bf16.

### L1 — b=8 : un poids lu une seule fois pour les 8 lignes, et décodé une seule fois
*Sources : sparkinfer (gemv.cu l. 856-895, PR #1081, PR #1132), sm120_gemm (tinymtc), vLLM-Moet
(GEMV MXFP8), q27 (fused weight sweep).*
- Vérifier dans chaque projection d'acvram à M=8 : (a) aucune boucle ligne par ligne cachée dans
  un chemin de repli (sparkinfer : 59 % du temps GPU à c16) ; (b) le décodage e2m1 et
  l'application des échelles ne sont pas refaits par ligne (1,82× pour 2 lignes chez sparkinfer) ;
  (c) aucune table e2m1 indexée dynamiquement (mémoire locale ; 21–46 % de bande passante).
- À partir de M ≈ 5–6, basculer sur une GEMV à cœurs tenseurs `mma.m16n8k16` avec déquantification
  dans les fragments B : coût plat de M=1 à 16 (sm120_gemm). Seuil d'aiguillage par forme.
- Replier la réduction split-K dans le noyau (13 % du coût à M=8 chez sm120_gemm).
- Budget cible *(calcul)* : q27 dépense ~5,6 ms hors projections pour un tour de ~8 positions ;
  qwen3.5-triton ~2,2 ms hors GEMV à b=1. Les 13 ms d'acvram laissent donc de l'ordre de 5 à 7 ms
  récupérables.
- Contrôle falsifiable : mesurer le temps d'une projection à M=1, 2, 4, 8 ; si t(8)/t(1) > ~1,3,
  le poids est relu ou re-décodé.

### L2 — b=8 : coût fixe des frontières de noyau (PDL, fusion, préchargement L2)
*Sources : pdl-megakernel-reconstruction, q27 (1 643 nœuds/tour), AlpinDale, mxfp6_sm120.*
- *(calcul)* ~2,3 µs par frontière même sous graphe CUDA (190 µs / 81 noyaux) ; à ~1 600 nœuds par
  pas, l'ordre de grandeur est de 1 à 3 ms. PDL a repris 83 % de ce trou chez tie-pilot-qxw.
- Déclencheur **après émission** des charges de poids du producteur ; attente au premier usage de
  l'activation, après avoir émis ses propres charges de poids.
- Fusions qui retirent des nœuds : résiduel + RMSNorm + quantification, porte RMSNorm GDN +
  préparation de l'entrée de projection (mxfp6_sm120), tête LM + argmax (AlpinDale,
  qwen3.5-triton).
- Précharger en L2 les poids de la phase suivante pendant l'attention et le GDN, pas pendant une
  GEMV (−8,7 % mesuré sinon).

### L3 — GDN : état en 16 bits et noyau de décodage fusionné
*Sources : HyperQwen (+37 %), sparkinfer (compaction fp32 → bf16), qwen3.5-triton (DeltaNet
fusionné 0,56 ms), q27 (0,78 ms de récurrence par tour), qwentin (~155 Mio/client), discussion
HF #51.*
- *(calcul)* Qwen3.8-27B : 48 couches × 48 têtes v × 128×128 × 4 o = ~151 Mo d'état fp32 par
  séquence (cohérent avec les ~155 Mio de qwentin) ; lu + écrit = ~302 Mo/pas/séquence, soit
  ~2,4 Go à b=8 ≈ **1,35 ms** à 1,79 To/s. En bf16/fp16 : ~0,7 ms gagnées à b=8, et double de la
  concurrence possible.
- Test d'équivalence : mesurer l'écart de logits sur de longues générations (l'erreur de l'état
  s'accumule), pas seulement la PPL courte.
- Un seul lancement par couche GDN pour post-projection + conv1d + récurrence + sortie.
- Vérifier qu'aucune garde de forme (type `num_v_heads == 8 * num_k_heads`) ne renvoie vers un
  chemin lent pour 48/16.

### L4 — préfill : une seule disposition de poids lue par la GEMV et par la MMA
*Sources : NInfer (W4A4 au préfill / A16 au décodage sur le même artefact), knivesysl/qwentin
(poids empaquetés dans la disposition de fragment), sm120_gemm (fragment = 2 octets e2m1 dans un
bloc d'échelle), net-snix (croisement à M≈128 : 5,5 ms contre 19,6 ms à M=16384), sparkinfer
(préfill CUTLASS sm120).*
- Supprimer le dépaquetage Marlin en choisissant une disposition que la GEMV de décodage et la
  GEMM `mxf4nvf4` du préfill lisent toutes deux sans conversion ; à défaut, dépaqueter une fois au
  chargement vers un second format (net-snix, +1× VRAM FP4 — incompatible avec 32 Go pour un
  27B + KV, donc à éviter).
- Au préfill, W4A4 natif (`mma…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) :
  ~3,6× la voie Marlin à M=16384 chez net-snix. Coût : quantifier l'activation en NVFP4 →
  mesurer la PPL préfill ; NInfer le fait en production.
- Détails à reprendre : ordre d'inclusion CUTLASS (`arch/config.h` avant `float_subbyte.h`),
  `EVICT_FIRST` via `.L2::cache_hint` sur le flux de poids, tuile 256×128×128 pour m ≥ 16k
  (sparkinfer).
- GDN au préfill par blocs WY/UT en espace log (sparkinfer, FlashQLA : 2,76× sur le noyau) — gain
  de bout en bout modeste à 8k (~3 %), fort sur les longs contextes (20,7 % du préfill à 4k chez
  sparkinfer avant correction).

### L5 — qualité NVFP4 : protéger attention, GDN, tête et embedding en FP8
*Sources : NInfer / unsloth (allocation mixte), imp (mesures par groupe), mxfp6_sm120 et qwentin
(échelles de qualité), fouroversix (4/6).*
- Allocation éprouvée en production chez NInfer et sparkinfer : **FP8 par ligne** pour les
  projections d'attention q/k/v/o, GDN qkv/z/out, embedding, lm_head et MLP des couches 56–63 ;
  NVFP4 pour les MLP 0–55 seulement. Coût en octets *(calcul)* : NInfer pèse 20,0 Gio contre
  17,9 Go en NVFP4 uniforme, et décode 71 t/s contre 96 t/s (sparkinfer) — ~25 % de vitesse b=1
  pour la qualité. À mesurer en PPL et KL avant décision.
- Mesures publiées : MAE logprob NVFP4 = 3,1× celle de FP8 ; top-1 NVFP4 85,8 contre FP8 95,9.
- Points fins d'imp : la tête MTP ne doit jamais être quantifiée ; lm_head NVFP4 = +0,99 % PPL
  pour +10,4 % de vitesse ; garder le q_proj Q+gate en source a *dégradé* la PPL de 1,5 % ;
  replier une échelle dans un RMSNorm `(1+g)` en bf16 perd de la précision.
- Gain gratuit à tester : 4/6 (choix max 4 ou 6 par bloc à la quantification), même format,
  aucun changement de noyau.
- Juger sur KL et sur les positions catastrophiques, pas sur la seule PPL (q27).

### L6 — hôte et service (4,8 ms)
*Sources : q27 (`docs/perf-next-2026-09-12.md`), vLLM-Moet, AlpinDale, imp.*
- Gabarit + tokenisation : cache BPE borné, ne pas ré-encoder le bloc système (38 → 12 ms chez
  q27, par requête).
- Dans le graphe : chercher les nœuds d'attente hôte (`cuStreamWaitValue32`, 1,2 ms/pas chez
  vLLM-Moet), les `cudaMemsetAsync` et les copies depuis mémoire paginable ; entrées du pas par
  `cudaMemcpyAsync` depuis un tampon épinglé, remises à zéro sur le GPU (AlpinDale).
- Cible de comparaison : q27 = 0,22 ms d'hôte par tour ; imp = 2,5 ms d'attente en file p50 à
  1 flux.

### L7 — secondaires
- Brouillon par blocs (DFlash2 / DSpark) : +22 % agrégés sur MTP chez q27 ; 3,2× annoncé pour
  gemma4-31B (lucebox). Hors des manques listés.
- Attention de vérification sans matérialiser le KV quantifié (Maharajahu : +49,8 % à 190k).
- FA2 accumulation f16 pour QKᵀ et PV (imp : +3–4 % préfill, +0,37 % PPL) ; FP4 dans l'attention
  réfuté.
- GEMM NVFP4 : tuiles CTA-majeures + entrelacement de colonnes (imp), stream-K pour la dent de
  scie des vagues (sm120_gemm).
