# poste4 — duck.ai (30/09 nuit), 2 questions, 3 modèles de raisonnement

Modèles : gpt-oss 120B, GPT-5.6 Luna, Gemma 4 31B (recherche Web activée
sauf mention contraire). Un chiffre n'est retenu que confirmé par deux
sources indépendantes ; sinon marqué « source unique » ou « désaccord ».

## Q1 — vLLM multimodal (Qwen3-VL, Gemma 3/4) : encodeur de vision avant profilage/KV, réservation d'activations

- **Luna** (sourcée GitHub, vllm.ai) : oui — dans V1, l'encodeur est
  exécuté/profilé avant le profilage du backbone texte et avant
  l'allocation KV (`profile_run() → profile_encoder_cache() →
  _dummy_run()`, `gpu_model_runner.py`). Le pic encodeur est donc inclus
  dans la mémoire restante pour le KV-cache, pas mesuré après. Entrées
  factices dimensionnées par `MultiModalBudget` sur l'item/taille donnant
  le plus de features. Budget matérialisé par des sorties factices
  `(encoder_budget, hidden_size)`. `--skip-mm-profiling` supprime ce
  profilage (laisse l'estimation à l'utilisateur, `vllm/config/multimodal.py`).
- **gpt-oss** (mêmes fichiers, corroboré) : même ordre confirmé ; ajoute le
  message de log au démarrage (« Encoder cache will be initialized with a
  budget of N tokens... ») et nomme `compute_encoder_budget` dans
  `encoder_cache_manager.py` pour le calcul du budget, et l'allocation d'un
  tenseur unique `(encoder_budget, hidden_dim)`.
- **Gemma 4 31B** : même conclusion générale, mais cite `vllm/worker.py`
  (`profile_model`) et `qwen2_vl.py` — **chemins non retrouvés par les deux
  autres modèles, probablement obsolètes ou inventés** (la structure réelle
  nommée par Luna/gpt-oss est `gpu_model_runner.py` +
  `encoder_cache_manager.py`). Gemma écartée sur l'attribution des
  fichiers, pas sur la conclusion générale.
- **Verdict** : accord total sur le fond (encodeur chargé et profilé AVANT
  le KV-cache, budget dimensionné sur l'image/les items maximaux). Fichiers
  fiables : `vllm/v1/worker/gpu_model_runner.py`,
  `vllm/v1/core/encoder_cache_manager.py`, `vllm/config/multimodal.py`
  (Luna + gpt-oss, convergents) ; écarter les chemins Gemma.

## Q2 — vLLM MoE NVFP4/Marlin : poids naturels libérés après reconditionnement, ou coexistence au pic ?

- **Luna** (sourcée vllm.ai) : coexistence au pic — Marlin alloue les
  tenseurs réordonnés avant de remplacer les poids naturels ; après
  `prepare_moe_fp4_layer_for_marlin(layer)` / `process_weights_after_loading`,
  vLLM supprime les poids/scales intermédiaires (`del layer.w13_weight,
  w2_weight`). La mémoire allouée redescend ensuite, mais le cache de
  l'allocateur CUDA (`memory_reserved`) peut rester haut. Fichiers :
  `vllm/model_executor/layers/quantization/modelopt.py`
  (`ModelOptNvFp4FusedMoE.process_weights_after_loading`),
  `.../utils/marlin_utils_fp4.py`.
- **gpt-oss** : même conclusion (coexistence temporaire au pic du
  reconditionnement, puis libération des originaux une fois le nouveau
  format installé) — source plus faible (RFC #54477, contexte reload RL,
  pas MoE NVFP4 spécifique).
- **Gemma 4 31B** : même conclusion, source `marlin.py` non vérifiée
  (probablement générique, pas le fichier NVFP4 MoE réel).
- **Verdict** : accord total des 3, aucun modèle écarté sur le fond — les
  deux copies **coexistent au pic mémoire** pendant le reconditionnement
  Marlin, les poids naturels sont **libérés après** (via `del` explicite
  côté vLLM). Retenir les fichiers Luna (les plus précis et vérifiés :
  `modelopt.py`, `marlin_utils_fp4.py`) ; la réserve CUDA
  (`memory_reserved`) peut rester haute même après libération logique — à
  ne pas confondre avec la mémoire allouée réelle si un budget est mesuré
  juste après le chargement.
