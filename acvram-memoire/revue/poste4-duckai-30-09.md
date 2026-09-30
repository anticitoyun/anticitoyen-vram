# poste4 — duck.ai (30/09), 4 questions, 3 modèles de raisonnement

Modèles : gpt-oss 120B, GPT-5.6 Luna, Gemma 4 31B (recherche Web activée sauf
mention contraire). Un chiffre n'est retenu que confirmé par deux sources
indépendantes ; sinon marqué « source unique » ou « désaccord ».

## Q1 — YALS/llama.cpp : ssm_conv1d (Qwen3.6/3.8 GDN) et Jinja « is sequence » (Nemotron)

- **gpt-oss** : ssm_conv1d corrigé par la PR llama.cpp #20411, builds après ce
  merge (ex. b8287+) ; Jinja « Unknown test: sequence » introduit en b8227,
  corrigé par le build précédent b8226 (ou plus récent). Sources : HF
  (unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF), GitHub (issue autoparser).
- **Luna** : contredit gpt-oss — l'échec ssm_conv1d persiste encore sur des
  builds récents (b9119/ef93e98d, mai 2026) selon deux discussions HF
  (unsloth Qwen3.6-35B/27B-MTP-GGUF) : ce n'est pas un seuil de version mais
  un GGUF MTP/UD mal exporté (branche mtp-clean requise). Pour Jinja, aucun
  commit/build YALS trouvé ; suggère llama.cpp 2026 (b10902+) ou `is iterable`
  en repli. Source : github.com/ggml-org/llama.cpp/discussions/22132.
- **Gemma 4 31B** : support Qwen3.5/3.6 GDN intégré vers janvier 2026
  (Qwen3_5TextModel, source HF) ; mais avertit que « missing tensor
  ssm_conv1d » vient souvent d'un bug d'export Unsloth qui tronque la
  dernière couche (GitHub, indépendant de la version serveur) — converge
  avec Luna. Jinja : pas de commit précis, recommande de suivre les commits
  jinja de llama.cpp.
- **Verdict** : 2/3 (Luna, Gemma) convergent sur « pas un simple seuil de
  version, cause = export GGUF défaillant » ; gpt-oss (seul à donner des
  numéros de PR/build précis, non recoupés) est **écarté comme source
  unique invérifiée** sur ce point. Aucun commit YALS propre trouvé par
  aucun modèle pour le Jinja — à vérifier directement dans le dépôt YALS.

## Q2 — kimi-code CLI : réessais sur HTTP 200 flux vide

- **gpt-oss** : 5 tentatives totales, backoff exponentiel 0,3 s → plafond
  ~5 s. Borne : `loop_control.max_attempts_per_step` dans
  `~/.kimi-code/config.toml` (1 = échec immédiat). Sources : GitHub (fix
  openai_legacy empty streaming), kimi.com/docs.
- **Luna** : 10 tentatives totales (9 réessais), backoff doublé plafonné à
  32 s, respecte `Retry-After`. Mêmes clés de config
  (`loop_control.max_attempts_per_step` / var d'env
  `KIMI_LOOP_MAX_ATTEMPTS_PER_STEP`), plus `KIMI_CODE_INFINITE_RETRY=1` pour
  suppression de la borne. Note une ancienne doc à 5 tentatives / 5 s max
  (= la version que gpt-oss a citée). Sources : kimi.com/docs, GitHub.
- **Gemma 4 31B** : **aucune documentation trouvée**, refuse de chiffrer —
  renvoie au comportement générique d'un client OpenAI-compatible.
- **Verdict** : désaccord gpt-oss (5, doc ancienne) / Luna (10, doc
  actuelle, plus détaillée et datée) — retenir **Luna** comme version
  courante, gpt-oss comme version historique nommée par Luna elle-même.
  Gemma n'invente pas : elle décline faute de source, ce qui est le
  comportement correct prévu par le contrat.

## Q3 — Gemma 4 31B : taille cache KV/jeton (bf16, int8), dimensionnement SWA vLLM/llama.cpp

- **Gemma 4 31B** (sans recherche Web) : ~192 Ko/jeton bf16, ~96 Ko/jeton
  int8 (48 couches, 8 têtes KV, dim 128 — chiffres non sourcés). vLLM et
  llama.cpp dimensionnent les couches SWA sur la fenêtre seule, pas le
  contexte entier (PagedAttention côté vLLM, cache circulaire côté
  llama.cpp) ; couches globales = contexte entier.
- **Luna** (sourcée, config HF) : 50 couches SWA (16 têtes KV × 256) + 10
  couches globales (4 têtes KV × 512) → 430 080 valeurs KV/jeton = 0,82 MiB
  bf16, 0,41 MiB int8. vLLM : allocation hybride, couches globales
  réservent tout le contexte, couches sliding seulement `sliding_window`
  (blocs hors fenêtre libérés). llama.cpp : `kv_cache_iswa` fait la même
  chose par défaut (SWA = fenêtre, global = contexte entier), sauf
  `--swa-full` qui force une réservation pleine. Sources : HF config Gemma
  4 31B, vllm.ai (Hybrid KV Cache Manager), GitHub (SWA KV cache
  compression).
- **gpt-oss** : ~168 Ko/jeton bf16, ~84 Ko/jeton int8 — et affirme que
  **llama.cpp ne partage pas entre couches et grossit avec le contexte
  entier pour les 60 couches**, contredisant Luna et Gemma (et l'
  implémentation connue `kv_cache_iswa`).
- **Verdict** : Luna la plus complète et sourcée (config réelle HF,
  mécanique vLLM/llama.cpp cohérente entre les deux). **gpt-oss écarté** :
  chiffres non recoupés et affirmation sur llama.cpp contredite par les deux
  autres modèles. Retenir les chiffres Luna : **0,82 MiB/jeton bf16,
  0,41 MiB/jeton int8** (630 080 valeurs KV), SWA = fenêtre only sur les
  deux moteurs (sauf `--swa-full`).

## Q4 — vLLM : reconditionnement Marlin (NVFP4 MoE) vs profilage mémoire/cache KV

- **gpt-oss** : reconditionnement (`process_weights_after_loading`) a lieu
  avant le profilage mémoire et l'allocation KV ; le pic mémoire EST compté
  dans `gpu_memory_utilization` (le journal vLLM sépare poids/reste, le
  « reste » inclut la surcharge de reconditionnement).
- **Luna** (sourcée fichiers vLLM) : confirme l'ordre — `load_model()`
  (donc le reconditionnement Marlin) s'exécute avant
  `determine_available_memory()` → `profile_run()` → allocation KV. Mais
  **contredit gpt-oss sur le pic** : `reset_peak_memory_stats()` intervient
  après le chargement, donc le pic transitoire du reconditionnement
  **échappe** au profilage et n'est pas capté par `gpu_memory_utilization`
  (seule l'empreinte résidente finale des poids l'est). Sources :
  `vllm/v1/worker/gpu_worker.py`, `vllm/model_executor/model_loader/base_loader.py`.
- **Gemma 4 31B** (sans recherche Web, fichiers non vérifiés en ligne) :
  même ordre (reconditionnement avant profilage/KV) ; même conclusion que
  Luna sur le pic (non compté, seule l'empreinte finale compte). Fichiers
  cités (`marlin.py`, `worker.py`) plausibles mais **non vérifiés par une
  recherche Web** — à confirmer avant citation.
- **Verdict** : ordre AVANT confirmé par les 3 (accord total). Sur le pic
  mémoire, **2/3 (Luna, Gemma) contre gpt-oss** : le pic transitoire du
  reconditionnement Marlin **n'est pas compté** dans
  `gpu_memory_utilization`, seule l'empreinte résidente finale des poids
  l'est. gpt-oss écarté sur ce point (source faible, citations numérotées
  sans fichier nommé).
