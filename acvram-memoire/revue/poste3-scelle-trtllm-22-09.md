# Scellé — cellules TRT-LLM vs acvram, Qwen3-Coder-30B-A3B (poste3, 22/09, AVANT mesure)

Scellé écrit AVANT toute mesure (REGLES § 4). poste2 exécute `scratchpad/trtllm-cellules-22-09/cellule.sh` après P3 / frontière / ABBA / E. Prédictions chiffrées, seuils, et tableau des différences → réglage ou note.

## Modèles et moteurs

- acvram : NVFP4 maison (bloc 16, échelles E4M3), cache KV **INT8**, graphes CUDA actifs (défaut ≥ 0.6), moteur 665eeacc (défaut sampler lent, revert).
- TensorRT-LLM 1.3.0rc15, backend PyTorch : checkpoint hub `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (ModelOpt, bloc 16 E4M3, cache KV **FP8** par défaut ; `exclude` = portes MoE `mlp.gate` de chaque couche + `lm_head`). sha des safetensors relevés en tête du TSV par cellule.sh.
- Même carte (RTX 5090, ACVRAM_CARTE=0), horloge relevée par fenêtre, llama-server voisin sur la 3080 Ti (présent dans toutes les cellules, non touché).

## Prédictions chiffrées (mon attente, à sceller) — glouton T=0, Qwen3-Coder-30B

| cellule | acvram 0.6.34 (mesuré) | vLLM (mesuré) | **TRT-LLM prédit** | issue nommée |
|---|---|---|---|---|
| décodage b=1 | 380,8 t/s | 290,6 | **380–470, centrale 420** | TRT-LLM ≥ acvram probable (moteur NVIDIA) ; si < 380, acvram devant à b=1 |
| décodage b=12 | 1 540 t/s | 1 596 | **1 550–1 850, centrale 1 700** | TRT-LLM ≥ vLLM probable ; comparabilité limitée par KV (voir note) |
| prefill pp2048 | 22 707 j/s | 21 054 | **24 000–34 000, centrale 29 000** | TRT-LLM probablement devant au prefill |

Ces prédictions ADMETTENT que TRT-LLM (moteur de référence NVIDIA) puisse dominer acvram : le but est une mesure honnête à 4 moteurs, pas de faire gagner acvram. Réfuté (prédiction trop basse) si TRT-LLM dépasse la borne haute ; réfuté (trop haute) s'il tombe sous la borne basse — dans les deux cas la valeur mesurée est publiée telle quelle.

## Seuils de rejet d'une fenêtre (protocole, hérité du banc 4 moteurs)

Fenêtre invalidée si : < 20 s ; écart-type > 5 % sur 3 relevés ; GPU non isolé (`CUDA_VISIBLE_DEVICES` non vide hors carte.sh) ; charge étrangère mesurée au-delà du seuil (load, autres process GPU). 6 fenêtres **intercalées A B B A A B** (A = acvram, B = TRT-LLM), horloge **médiane par fenêtre** écrite. Débit retenu = médiane des fenêtres valides par moteur.

## Tableau des différences → réglage ou NOTE (liste de contrôle 3.4)

| différence | acvram | TRT-LLM | traitement |
|---|---|---|---|
| cache KV | INT8 | FP8 | **NOTE** — `int8` n'existe pas côté TRT-LLM PyTorch 1.3 (`KvCacheConfig(dtype=)` ∈ {auto, fp8}) ; non égalisable, porté sur la cellule |
| exclusions de quantif | SENSITIVE_SUFFIXES : `layernorm/norm/_norm`, `conv1d*`, `a_log`, `dt_bias`, `mamba.*`, `e_score_correction_bias`, `.a.weight` (+ q_b/k_b/v_b bf16) | portes MoE `mlp.gate` (48 couches) + `lm_head` | **NOTE** — jeux différents ; les deux gardent normes et portes de routage hors quantif ; listé des deux côtés, pas « aucun » |
| graphes CUDA | actifs (défaut) | **actifs** `CudaGraphConfig(batch_sizes=[1,12], enable_padding=True)` | **RÉGLAGE** — activés des DEUX côtés (ne pas couper) |
| échantillonnage | glouton T=0 | glouton `SamplingParams(temperature=0.0)` | **RÉGLAGE** — glouton des deux côtés |
| contexte / émission | même invite, même `max_tokens` | idem | **RÉGLAGE** — longueurs identiques (prefill pp2048 ; décodage même nombre de jetons émis) |
| bloc de quantif | 16, E4M3 | 16, E4M3 | identique |
| `max_num_tokens` / godets | godets acvram | `max_batch_size=12`, `max_num_tokens` fixé | **RÉGLAGE** — mêmes godets de lot {1, 12} |
| carte / horloge | 5090, `-lgc` relevé | 5090, horloge relevée | **RÉGLAGE** — même carte, horloge médiane par fenêtre à l'en-tête |

## En-tête TSV de publication (rejouable)

`moteur  modele  sha_modele  cellule(b1|b12|prefill)  t/s  J/jeton  horloge_med_MHz  temperature  charge_load  ctx  emis  date  duree_s  fenetre(A|B + rang)  kv_dtype  graphes_cuda`

## Ce qui reste à poste2 (exécution)

Lancer `cellule.sh` sous carte.sh mesure après ses pièces en cours. Le script prend le verrou, `source env.sh` (MPI), sert acvram puis TRT-LLM à tour de rôle (A B B A A B), envoie la charge b=1 / b=12 / prefill ≥ 20 s, relève t/s + horloge médiane + énergie NVML, écrit le TSV avec l'en-tête ci-dessus. Le J/jeton net suit le protocole Q2.5 (soustraction du repos, intégration sur la fenêtre). Réutilise le harnais `outils/gpu/mesure/banc-4moteurs.py` (un moteur à la fois) ; si TRT-LLM n'y est pas encore branché, l'ajouter est une note d'exécution, pas une décision de scellé.
