# Duel MLA NVFP4 vs vLLM, GLM-4.7-Grande-Heretic-42B — parqué

Océane, 14/09/2026 soir. Ordre de Sage/Jérôme (revue/sage-strategie-14-09.md) :
duel apparié acvram (MLA NVFP4 sm_120) contre vLLM sur ce modèle, décodage
b=1/4/12 + J/jeton — créneau où acvram gagne ×1,40 (bead 6wa, Laurine) et
qu'on veut rendre défendable.

## Prérequis manquant, vérifié avant de lancer

Aucune source bf16/safetensors pour ce modèle sur le poste. Le manifeste
de notre propre conversion NVFP4 (`acvram_manifest.json`, champ
`model.name`) nomme lui-même sa source :
`GLM-4.7-30B-A3B-20-2-Heretic-30B-A3B-Q4_K_M.gguf` — déjà un GGUF Q4_K_M,
pas un bf16 (`torch_dtype: bfloat16` dans le manifeste décrit l'architecture
native, pas un fichier de poids qu'on a). Cherché sous
`/mnt/4TO_SATACMR_2022/Modeles` (tout sous-dossier `GLM*4.7*`, tout
`.safetensors`) : rien. Seuls existants : notre nvfp4
(`models_acvram/GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4`) et le GGUF de
Laure (`models_gguf/GLM-4.7-Grande-Heretic-42B-Q4_K_M`).

## Hypothèse (a) essayée, réfutée en < 10 min (0 min de carte)

Jérôme : un seul essai borné, `vLLM --quantization gguf` sur ce MÊME
fichier GGUF (duel le plus apparié — même source, deux moteurs), 1 h / 3
hypothèses max.

`/opt/ia/vLLM` (0.29.0, la même version que le duel A2 de Manon) —
**aucun support GGUF dans cette installation** : pas de module
`vllm.model_executor.layers.quantization.gguf`, et une recherche
insensible à la casse de "gguf" dans tout le paquet ne trouve que 3
mentions incidentes (des commentaires dans `lora/layers/utils.py`,
`exaone_moe.py`, `qwen2_moe.py`) — aucun chemin de chargement réel.
Vérifié sans GPU, sans carte, avant tout chargement : `--quantization
gguf` aurait échoué au tout premier import, pas à l'inférence.

## Verdict

**Parqué.** Pas de source bf16 sur le poste ; le GGUF existant n'est pas
chargeable par vLLM 0.29.0 (pas de support GGUF dans ce build). Reste :
télécharger la source bf16 (~80 Go, décision utilisateur, pas prise ici)
si le duel reste voulu — ou changer de version de vLLM si une version
avec support GGUF existe et charge correctement un MoE (pas vérifié,
hors de la borne des 3 hypothèses posée par Jérôme).

# Reprise du 14/09 soir : GLM-4.7-Flash, apparié (décision utilisateur)

Décision de l'utilisateur : même modèle MLA des deux côtés —
`zai-org/GLM-4.7-Flash` (glm4_moe_lite, 30B-A3B, 47 couches, 64 experts
top-4 + 1 partagé, MLA q_lora 768 / kv_lora 512 / nope 192 / rope 64 /
v 256, 20 têtes, vocab 154 880). bf16 (62 Go) en téléchargement vers
`/mnt/4TO_SATACMR_2022/Modeles/GLM-4.7-Flash-bf16` (unité hf-glm47flash) ;
Manon convertit en NVFP4 (srcbf16) pour acvram.

## Préparation du bras vLLM (Laurine, hors carte)

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

### Chargement à sec vLLM + GLM-4.7-Flash-NVFP4 (14/09, 20h38-20h45, carte.sh, chargement seul)

1. `attention_config={"backend": "TRITON_ATTN"}` (celui du Coder) → `ValueError:
   MLA not supported` : pour un modèle MLA, laisser vLLM choisir
   (`BANC_ATTN=auto` dans `ncu_vllm_decode12.py` ; `banc_decode_vllm_glm.py` ne
   force rien).
2. **TRITON_MLA échoue sur sm_120 à la capture des graphes** :
   `triton.runtime.errors.OutOfResources: shared memory, Required: 102400,
   Hardware limit: 101376` dans `_decode_grouped_att_m_fwd`
   (`v1/attention/ops/triton_decode_attention.py:534`). Cause lue : Lk = 576
   (kv_lora 512 + rope 64, dims DeepSeek/GLM) → BLOCK_DMODEL 512, BLOCK_DPE 64,
   `num_stages = 2` ; la garde `num_stages = 1` n'existe que pour
   `BLOCK_DMODEL >= 1024` (:530), écrite pour H100 (227 Ko de shared) — la 5090
   n'a que 99 Ko. Aucun réglage utilisateur ne l'évite (KV fp8 ou bf16 : mêmes
   tuiles).
3. **Correctif d'expérience, borné au processus** (`BANC_MLA_STAGES1=1` :
   la même règle dès 512, par `exec` du source de la fonction) : chargement OK
   — poids 18,04 Gio en 83 s, `VLLM_CUTLASS` NvFp4 MoE, KV cache 5,98 Gio
   (237 328 jetons), graphes PIECEWISE 51 + FULL 35 capturés, 4 pas de
   décodage b=12 exécutés (`NCU_OK 12 4`). Un correctif d'installation
   (`/opt/ia/vLLM`, 1 ligne) reste une décision utilisateur ; sans lui vLLM
   0.29 ne décode PAS un MLA aux dimensions DeepSeek sur RTX 50 — fait à
   verser au duel (le concurrent a besoin d'un patch pour jouer).
4. Pendant la manche, `carte.sh` a relevé power.limit 600 → 400 W (réglage
   d'un autre passage) : sans effet sur un chargement à sec.

Prochaine commande (duel, quand le srcbf16-nvfp4 de Manon existe) :
`BANC_MLA_STAGES1=1 outils/carte.sh /opt/ia/vLLM/.venv/bin/python
outils/banc_decode_vllm_glm.py` — à condition d'y porter le même correctif
d'expérience (fait : le banc lit `BANC_MLA_STAGES1`).
