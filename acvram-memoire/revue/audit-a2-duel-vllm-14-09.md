# Audit A2 — premier duel vLLM/TabbyAPI depuis la réinstallation

Manon, 14/09/2026. Consigne de Jérôme : valider vLLM et TabbyAPI sur
Qwen3-Coder-30B-A3B, régime pp2048, même NVML power.draw.instant (idle
soustrait), même fenêtre, contre le chiffre du jour d'acvram (16 938 j/s,
Laurine, cp.async).

## Modèle vLLM

Aucun checkpoint NVFP4 local ne correspondait exactement à
Qwen3-Coder-30B-A3B pour vLLM (seul `Qwen3.6-27B-NVFP4`, un autre modèle,
existait dans `models_vllm/`). **Téléchargé avec accord explicite de
l'utilisateur** (18,1 Gio) : `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`
(NVIDIA ModelOpt, base `Qwen/Qwen3-Coder-30B-A3B-Instruct`, le même modèle
de base que notre conversion acvram) → `models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4/`.

## Dénominateur commun (Laurine, 14/09)

Ma première mesure était fausse (cache de préfixe : invite identique à
chaque répétition, `enable_prefix_cache` par défaut → un pas de ~27 ms
quelle que soit L, débit gonflé). Laurine a pointé son script canonique
(`outils/banc_prefill_chaud.py`, branche laurine ea98f36, 16 938 j/s) et
son dénominateur : `L / durée d'un generate(max_tokens=1) complet`,
synchronisé aux deux bouts, invite **différente** à chaque répétition,
cache de préfixe désactivé, 2 passes de chauffe + 7 répétitions, médiane.
`outils/banc_prefill_vllm.py` (ce commit) reproduit ce protocole via
`vllm.LLM.generate()` en offline (pas de serveur HTTP — évite de mesurer
autre chose, comme `vllm bench latency` l'aurait fait selon Laurine).

## Trois obstacles d'infrastructure (« depuis la réinstallation »)

1. **FlashInfer absent du venv `/opt/ia/vLLM/.venv`** — `RuntimeError:
   FlashInfer backend is not available` au premier appel réel (backend
   choisi automatiquement). Pas réparé (installation du paquet hors de
   mon périmètre décidé ici) — contourné.
2. **FLASH_ATTN refuse le cache KV FP8** sur notre 5090 (sm_120) :
   `ValueError: ... FP8 KV cache requires FA3 on SM90 or FA4 on SM100`.
   Contourné par `attention_config={"backend": "TRITON_ATTN"}`.
3. **`max_model_len` par défaut (262 144, contexte max du modèle)**
   réservait plus de cache KV (12 Gio) que de VRAM libre (6,43 Gio) pour
   un seul prefill de 2048 jetons. Fixé à 4096, comme
   `banc_prefill_chaud.py` côté acvram.

Aucun de ces trois n'est un défaut d'acvram — tous trois sont propres à
l'installation vLLM de ce poste, pas mesurés avant aujourd'hui (« premier
duel depuis la réinstallation », Jérôme).

## Piège de puissance trouvé sur ma propre mesure

Envelopper tout le sous-processus vLLM (chargement du modèle + boucle de
mesure) dans un seul relevé NVML donnait 3 W nets (67 W fenêtre, 64 W
idle) — implausible pour du calcul réel sur une 5090. Cause : le
chargement (~30-60 s, quasi-idle côté GPU pendant la lecture des poids)
domine la fenêtre et dilue la médiane vers l'idle. Corrigé en déplaçant
l'échantillonnage NVML **à l'intérieur** de `banc_prefill_vllm.py`,
autour de la seule boucle de mesure (2 chauffes + 7 répétitions) — la
puissance de charge, elle, reste correcte pour `banc_prefill_chaud.py`
côté acvram (mesurée de l'extérieur, chargement apparemment plus court
avec les noyaux déjà en cache).

## Résultat

| moteur | pp2048 (j/s) | σ | ms/pas | puissance idle | puissance nette |
|---|---|---|---|---|---|
| acvram (MMA, S=4) | 17 111 | 37 | 119,7 | 17 W | 78 W |
| vLLM (TRITON_ATTN) | 34 788 | 1 441 | 58,9 | 64 W | 99 W |

**vLLM ≈ 2,03× plus rapide en prefill brut sur ce régime**, avec une
dispersion (σ=1441, ~4 % du débit) nettement plus large que la nôtre
(σ=37, ~0,2 %) — cohérent avec l'observation de Laurine du 13/09 sur le
duel llama.cpp (notre dispersion 22-31 % contre 1,5 % pour l'adversaire) :
un adversaire mieux industrialisé peut être plus rapide ET plus stable,
ou l'un des deux seulement — ici, plus rapide mais moins stable.

**Réserve sur l'idle** : 17 W (acvram) contre 64 W (vLLM) pour la même
carte, mesurés à des instants différents — la 5090 partage un bus PCIe
x8/x8 avec la 3080 Ti et d'autres sessions tournaient en parallèle
(carte.sh attendue avant chaque mesure) ; l'écart d'idle n'est peut-être
pas structurel, à vérifier si la comparaison doit trancher plus finement
que ×2.

## Addendum 14/09 — profil noyau et décodage à 12 séquences

Suite de Jérôme après le premier résultat : où est l'écart, et l'objectif
du projet (débit/énergie au décodage concurrent).

### Profil noyau, un pp2048

`torch.profiler` DEPUIS le processus appelant ne voit rien (vLLM exécute
l'inférence dans un processus séparé, `EngineCore`, via multiprocessing —
un profiler côté appelant ne capte que l'attente RPC,
`cudaDeviceSynchronize` 84 ms, ZÉRO noyau CUDA — piège trouvé en le
lançant). Corrigé avec le mécanisme intégré de vLLM
(`profiler_config={"profiler": "torch", ...}` + `llm.start_profile()`),
qui fait tourner le profiler DANS l'EngineCore. `outils/profil_vllm_pp2048.py`.

Total noyaux CUDA agrégé pour UN pp2048 : **49,4 ms** (contre notre pas
complet ~121 ms avant cp.async, 38 ms de noyau MMA seul depuis). Dix
premiers, par temps CUDA propre :

| ms | % | noyau |
|---|---|---|
| 14,31 | 29,0 | GEMM groupée CUTLASS FP4 (`GroupProblemShape`, block-scaled E2M1/UE4M3) — l'équivalent de notre MMA |
| 13,80 | 27,9 | `kernel_unified_attention` (Triton) |
| 4,61 | 9,3 | multiplication élémentaire bf16 (SiLU·up, la glue de porte) |
| 4,03 | 8,2 | `shuffleInputRowsKernel` (rassemblement par expert) |
| 3,28 | 6,7 | GEMM CUTLASS FP4 (non groupée — projections denses probablement) |
| 1,67 | 3,4 | `cvt_fp16_to_fp4` (quantification d'activation) |
| 1,50 | 3,0 | `reduce_kernel` (somme) |
| 1,45 | 2,9 | `cvt_fp16_to_fp4` (seconde instance) |
| 0,89 | 1,8 | `triton_red_fused_3` |
| 0,54 | 1,1 | `compute_arg_sorts` (tri du routage) |

**Leur GEMM groupée FP4 seule (14,31 ms) est déjà plus rapide que notre
noyau MMA seul (38 ms, post-cp.async)** — l'écart n'est donc pas
uniquement dans la glue Python/lancements (notre piste habituelle), il
est aussi dans le noyau GEMM lui-même. Face au profil de Laurine
(MMA 38 ms, glue, int8 dense 30 ms, flash 11 ms sur 121 ms), vLLM répartit
différemment : GEMM+attention = 28,1 ms (57 % du total), glue+routage+
quantification = 21,3 ms (43 %) — proportion de glue comparable, mais sur
un total 2,4× plus petit.

### Décodage à 12 séquences (objectif du projet)

Même protocole que Laure (`outils/gpu/mesure/banc-horloge-decodage.py`) :
12 séquences, contexte 2048, 200 jetons décodés chacune, énergie NVML
monotone (compteur, pas une moyenne de puissances — `energie.py`), repos
mesuré avant (8 s), net = brut − repos×durée. `outils/banc_decode_vllm.py`.

| moteur | t/s agrégé | J/jeton net | horloge SM (MHz) | bridage pendant la fenêtre |
|---|---|---|---|---|
| acvram (Laure, référence) | 568,6 | 0,601 | — | — |
| vLLM | 1198,4 | 0,202 | 2572-2827 | puissance (plafond 400 W atteint) |

**vLLM ≈ 2,11× le débit ET ≈ 2,97× plus efficace en énergie par jeton.**
Les deux mesures tournent sous le même bridage de puissance (400 W,
`nvidia-smi`), donc le bridage n'explique pas l'écart à lui seul — au
contraire, vLLM atteint un débit bien supérieur MALGRÉ le même plafond,
ce qui suggère un travail utile par watt structurellement meilleur (noyau
GEMM FP4 plus efficace, confirmé par le profil ci-dessus).

**Réserve** : `watts_repos` mesuré ici (66,9 W) reste élevé pour un
« repos » de 5090 — comme pour pp2048, la carte partage un bus PCIe
x8/x8 avec la 3080 Ti et d'autres sessions tournaient en parallèle
malgré `carte.sh` ; l'écart net (brut − repos) reste correct tant que le
repos est mesuré JUSTE AVANT la fenêtre (ce qui est le cas), mais un
« vrai » repos (carte seule, aucune autre session active) donnerait un
J/jeton net légèrement différent des deux côtés — pas de raison de penser
que ça inverserait le facteur ×3.

## Addendum 14/09 (suite) — profil du décodage et configuration exacte

Deux dernières questions de Jérôme avant TabbyAPI.

### Configuration exacte (logs de démarrage + `hf_quant_config.json`)

- **Backend MoE** : `VLLM_CUTLASS` NvFp4 (choisi parmi
  `FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTEDSL_BATCHED,
  FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION` — les
  quatre premiers (FlashInfer) écartés car le paquet est absent du venv,
  voir plus haut). `MoEPrepareAndFinalizeNoDPEPModular` (pas de
  parallélisme expert par données).
- **Activations en entrée du MoE : FP4**, pas FP8 — confirmé par les
  noyaux `cvt_fp16_to_fp4` dans les deux profils (pp2048 ET décodage) :
  vLLM quantifie l'activation bf16 en E2M1 juste avant chaque GEMM
  groupée, donc **W4A4 comme nous**, pas W4A8/W4A16.
- **KV cache** : `fp8_e4m3` (`kv_cache_quant_algo` du checkpoint
  ModelOpt). C'est ce format qui a fait échouer FLASH_ATTN (exige
  FA3/SM90 ou FA4/SM100, absents sur sm_120) — TRITON_ATTN, retenu, l'
  accepte.
- **Attention** : `TRITON_ATTN` (forcé, FlashInfer absent).
- **Quantification** (`hf_quant_config.json`) : `NVFP4`, `group_size=16`
  (identique au nôtre) ; le routeur (`mlp.gate`) et `lm_head` sont
  **exclus** de la quantification sur les 48 couches — comme notre
  routeur/attention restés en A16.
- **Graphes CUDA** : `cudagraph_mode=FULL_AND_PIECEWISE`,
  `cudagraph_capture_sizes=[1,2,4,8,16,24,32,...,512]` (51 tailles,
  jusqu'à 512 séquences/jetons batchés), `max_cudagraph_capture_size=512`.

### Profil du pas de décodage, 12 séquences (vis-à-vis du 569 t/s / 1198 t/s)

`outils/profil_vllm_decode12.py`, 8 pas de décodage profilés (au lieu des
200 de la mesure d'énergie — assez pour un profil représentatif, trace
plus légère). Total noyaux CUDA agrégé : **115,09 ms pour 8 pas**
(≈ 14,4 ms/pas pour 12 séquences, cohérent avec 1198 t/s mesuré : 12
jetons produits toutes les ~10 ms de calcul GPU, le reste étant la
plomberie CPU/lancement). Dix premiers, par temps CUDA propre :

| ms (8 pas) | % | appels | noyau |
|---|---|---|---|
| 47,62 | 41,4 | 864 | GEMM groupée CUTLASS FP4 (MoE, `GroupProblemShape`) |
| 9,19 | 8,0 | 768 | GEMM CUTLASS FP4 (non groupée) |
| 7,72 | 6,7 | 864 | `shuffleInputRowsKernel` (rassemblement par expert) |
| 5,47 | 4,8 | 96 | GEMM CUTLASS FP4 (troisième forme, tuile différente) |
| 5,27 | 4,6 | 432 | `kernel_unified_attention` (Triton) |
| 3,68 | 3,2 | 432 | `reduce_kernel` (somme) |
| 3,51 | 3,1 | 343 | **`cutlass_80_wmma_tensorop_bf16`** — noyau bf16 non quantifié, probablement le routeur/`lm_head` exclus de NVFP4 |
| 3,47 | 3,0 | 96 | `cvt_fp16_to_fp4` |
| 2,58 | 2,2 | 96 | `cvt_fp16_to_fp4` (seconde forme) |

**La GEMM groupée MoE domine largement au décodage (41,4 %, contre 29 %
au prefill)** — cohérent avec le régime : au décodage, chaque séquence
n'active que top-k experts sur 1 jeton, le ratio calcul/lancement est
pire, donc le noyau qui fait le plus de travail utile pèse relativement
plus lourd. Le noyau bf16 WMMA (ligne 7) confirme que le routeur reste en
précision pleine, comme annoncé par `hf_quant_config.json`.

## TabbyAPI (EXL3 4.0bpw)

`outils/banc_tabbyapi.py` : lance le serveur lui-même (`main.py
--model-name ... --max-seq-len ... --cache-size ...`), l'interroge en
HTTP, l'arrête — le verrou carte.sh enveloppe tout le script, pas
seulement la mesure. Trois obstacles, tous des détails d'API propres à
TabbyAPI (aucun ne remet en cause acvram ni vLLM) :

1. `/v1/completions` n'accepte qu'un `prompt` **string**, pas de jetons
   bruts comme vLLM/acvram — aller-retour par `/v1/token/decode` pour
   garder la même formule d'invite numérique ; le compte réel de jetons
   après retokenisation est lu dans `usage.prompt_tokens` du serveur
   (jamais supposé égal à L — la retokenisation du texte décodé dérive
   légèrement, 2127-2157 jetons mesurés pour un L visé de 2048).
2. `max_seq_len`/`cache_size` doivent être des multiples de 256
   (PAGE_SIZE d'exllamav3) — `AssertionError` sinon.
3. `usage` est `null` par défaut en non-streaming ; il faut
   `stream_options.include_usage: true` dans la requête MÊME sans
   streaming, sinon `AttributeError` en lisant `usage.prompt_tokens`.

Chargement du modèle (30B, EXL3 4.0bpw) : 187,8 s.

### Résultat

| moteur | pp2048 (j/s) | décodage 12 séq (t/s) | J/jeton net (décodage) |
|---|---|---|---|
| vLLM | 34 788 | 1 198,4 | 0,202 |
| acvram | 17 111 | 568,6 | 0,601 |
| **TabbyAPI** | **8 671** | **135,9** | **1,026** |

**TabbyAPI est nettement le plus lent des trois sur les deux axes** —
×2 plus lent qu'acvram en prefill, ×4,2 plus lent en décodage concurrent,
×1,7 moins efficace en énergie. Hypothèses plausibles, non vérifiées ici
(hors périmètre de cette mesure) : exllamav3/TabbyAPI est pensé pour un
usage mono-utilisateur (log de démarrage : « Disabling GPU split because
one GPU is in use »), pas pour la concurrence à 12 séquences que ce banc
impose ; le format EXL3 (quantification par couche, GPTQ-like) n'a pas
d'équivalent du noyau MoE groupé CUTLASS FP4 de vLLM ni du noyau MMA
natif d'acvram.

**Réserve** : mesuré sous le même bridage 400 W et une seule passe (pas
de jumelles) — un TabbyAPI mal configuré pour ce cas d'usage précis
(concurrence élevée) donnerait le même chiffre qu'un TabbyAPI
structurellement plus lent ; cette mesure ne les distingue pas.

## Bilan des trois moteurs

vLLM devant sur toute la ligne, acvram au milieu, TabbyAPI loin derrière
— dans CE régime (Coder-30B, NVFP4/FP4 pour les deux premiers, EXL3 pour
le troisième, 12 séquences concurrentes, RTX 5090 bridée 400 W). Les
noyaux FP4 CUTLASS de vLLM (GEMM groupée MoE, GEMM dense) sont plus
rapides que notre MMA natif malgré un travail équivalent (même format,
même bloc-échelle) — piste à approfondir côté acvram si le chantier le
justifie (hors périmètre de cet audit).
