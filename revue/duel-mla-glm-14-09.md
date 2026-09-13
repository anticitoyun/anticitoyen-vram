# Duel MLA apparié : GLM-4.7-Flash, acvram vs vLLM (14/09)

Décision de l'utilisateur : même modèle MLA des deux côtés —
`zai-org/GLM-4.7-Flash` (glm4_moe_lite, 30B-A3B, 47 couches, 64 experts
top-4 + 1 partagé, MLA q_lora 768 / kv_lora 512 / nope 192 / rope 64 /
v 256, 20 têtes, vocab 154 880). bf16 (62 Go) en téléchargement vers
`/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16` (unité hf-glm47flash) ;
poste2 convertit en NVFP4 (srcbf16) pour acvram.

## Préparation du bras vLLM (poste4, hors carte)

Lu dans `/opt/ia/vLLM` 0.29.0 (venv `/opt/ia/vLLM/.venv`) :

* **Architecture** : `Glm4MoeLiteForCausalLM` → `glm4_moe_lite`
  (`model_executor/models/registry.py:121`) ; MTP séparé
  (`glm4_moe_lite_mtp`), inactif sans spéculation.
* **MLA sur sm_120** : la liste des backends MLA pour `major == 12` est
  `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` (`platforms/cuda.py:129-133`) ;
  FlashInfer absent du venv, le sparse ne concerne pas GLM → **TRITON_MLA,
  seul et automatique** (ne PAS passer `attention_config`, qui vaut pour
  l'attention non-MLA). Il accepte le KV fp8 sur SM89+
  (`v1/attention/backends/mla/triton_mla.py:132-137, 233-244`) et les
  graphes CUDA en décodage (`_cudagraph_support =
  UNIFORM_SINGLE_TOKEN_DECODE`, :53) — même régime que le duel Coder
  (TRITON_ATTN + graphes).
* **`--quantization fp8` dynamique sur le bf16 : hors carte.** 30B de
  paramètres en fp8 = ~31 Go de poids pour 31,36 Gio de VRAM, avant KV et
  graphes. Le bras vLLM ne peut pas être « le bf16 quantifié à la volée ».
* **Checkpoint retenu : `GadflyII/GLM-4.7-Flash-NVFP4`** (HF, 20,4 Go,
  Apache-2.0, 16 k téléchargements) — compressed-tensors
  `nvfp4-pack-quantized`, bloc 16, échelles E4M3, **experts et MLP dense en
  E2M1, attention MLA / lm_head / routeur / embeddings en bf16** (calibration
  128 échantillons, tous les experts ; MMLU-Pro −1,3 pt vs bf16 contre −8,0
  pour un FP4 uniforme). C'est le format le plus proche du nôtre (nos
  experts NVFP4, nos projections MLA en int8 promu) : le duel compare deux
  NVFP4, pas un NVFP4 à un fp8. Chemin vLLM : scheme
  `CompressedTensorsW4A4Nvfp4` (linéaires) + `compressed_tensors_moe_w4a4_nvfp4`
  → `select_nvfp4_moe_backend` → `VLLM_CUTLASS` sur sm_120, comme le
  ModelOpt du Coder. Alternatives moins appariées : `cyankiwi/GLM-4.7-Flash-AWQ-4bit`
  (compressed-tensors W4A16), `QuantTrio/GLM-4.7-Flash-AWQ`.
  → **à télécharger (20,4 Go) vers
  `/mnt/4TO_SATACMR_2022/Modeles/models_vllm/GLM-4.7-Flash-NVFP4`** (1,2 To
  libres) ; décision utilisateur (téléchargement), pas la mienne.
* **Banc** : `outils/banc_decode_vllm_glm.py` — mêmes dénominateurs que
  `banc_decode_vllm.py` (jetons décodés / durée, compteur NVML monotone,
  repos 8 s, net = brut − repos×durée), `CUDA_VISIBLE_DEVICES=0` posé dans
  le script (5090 seule), b = 1 / 4 / 12, fenêtre ≥ 20 s (lots de 1 024
  jetons par séquence enchaînés jusqu'à 20 s, prefill 256 par lot publié),
  KV fp8 par défaut (`BANC_KV=auto` pour bf16), TRITON_MLA automatique,
  graphes actifs. `fenetre_valide` dans le RESULTAT.

Reste, dès que les fichiers sont là : chargement à sec sous `carte.sh`
(chargement seul : backend annoncé dans le log, `Using TRITON_MLA`,
`VLLM_CUTLASS` NvFp4 MoE, taille du KV, capture des graphes), pas de mesure.
