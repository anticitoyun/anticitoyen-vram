# Verdict — TensorRT-LLM sur Coder-30B : charge, décodage b=1/12, prefill 2048

instrument : `trtllm-bench throughput` (P3) ; `scratchpad/decode-trtllm-16-09.py` (fenêtre ≥ 20 s + 7 passes courtes, `energie.py`) ; `scratchpad/prefill-trtllm-16-09.py` (7 rép., médian) — journaux `scratchpad/trtllm-coder-16-09/`
commit : arbre laure ddf1bc5 + bras sous `__main__` (commit suivant) ; TRT-LLM 1.3.0rc15, torch 2.10.0+cu130
régime : converti `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (ModelOpt 0.33.1, NVFP4 g16, **KV fp8 dans le checkpoint** → `kv_cache_dtype=auto` = FP8), backend PyTorch, attention TRTLLM, MoE `AUTO`, graphes CUDA, blocs non réutilisés, KV 0,85, ctx 2048, invite 256, 1 024 jetons/séquence, une carte, plafond 400 W
scellé : P3a charge (70 %) · D1 b=12 ∈ [0,8 ; 1,3]× vLLM (1 150-1 870 t/s, J 0,21-0,34) · D2 b=1 ≥ 0,9× vLLM · P-1 prefill 2048 ≥ 15 000 j/s
mesuré : P3a **tenue** (1 502 t/s de sortie à 12 concurrents, `p3-bench.json`) · D1 **réfutée par le haut** : b=12 **2 104,7 t/s** (×1,46 vLLM), **0,177 J/jeton brut** (0,65× vLLM), meilleure passe courte 1 928 t/s · D2 tenue : b=1 **234,9 t/s** (×1,19), 1,481 J brut (×1,08 — plus cher que vLLM) · P-1 tenue : prefill 2048 **55 419 j/s** (37,0 ms, σ 278)
verdict : **TensorRT-LLM sert le NVFP4 MoE sur sm_120 et mène le comparatif sur Coder-30B : b=12 ×1,46 vLLM et ×3,3 acvram (14/09) en débit, 0,65× vLLM en J/jeton ; à b=1 il gagne en débit mais perd en énergie.** La seconde moitié de la prédiction Sage §4 (« ne sert pas NVFP4 MoE sur sm_120 ») est réfutée ; la première (0,9-1,3× vLLM en J) l'est aussi, par le bas (0,65×).

## Tableau (Coder-30B, mêmes dénominateurs que `energie-brute-faible-lot-14-09`)
    b    TRT-LLM t/s   TRT-LLM J/j brut   vLLM t/s (14/09)   vLLM J/j   acvram t/s (14/09)   acvram J/j
    1      234,9           1,481              196,7           1,373         233,0             1,430
    12   2 104,7           0,177            1 437,9           0,272         630,6             0,619
    prefill 2048 : TRT-LLM 55 419 j/s (37,0 ms) — vLLM Coder non mesuré ici (GLM : 26 732), acvram à apparier.

## Ce qui borne la lecture
- Les lots TRT-LLM font 1 024 jetons/séquence (prefill 256 = 20 % du lot) contre 3 836 chez vLLM le 14/09 (6 %) : le prefill pèse PLUS dans la fenêtre TRT-LLM ; l'écart ×1,46 est donc une borne basse, pas un chiffre apparié au jeton près. Le bras vLLM du comparatif sera refait avec ce même script (`BANC_JETONS=1024`).
- b=12 : `invalidations: bridage pendant la fenêtre : puissance` (372 W moyens, plafond 400 atteint) — la carte est bridée par choix de l'utilisateur ; publié tel quel.
- b=1 : le repos NVML vaut 67,7 W ici contre 89,4 W au 14/09 (vLLM) : les J bruts ne se comparent qu'à ±0,1 J/jeton ; le net (1,193 J) n'a pas de pendant vLLM.
- Prefill : 3 relevés de puissance sur 0,33 s → watts sans valeur (comme le bras vLLM) ; seul le temps compte.
- KV fp8 imposé par le checkpoint : vLLM 14/09 tournait en KV auto (bf16) — le comparatif final aligne les deux (§3 : même converti, KV explicite dans l'en-tête).
- Le même converti existait déjà : `/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4` (Manon, même producteur ModelOpt) — mon téléchargement de 17 Go est un doublon sur `/mnt/AI_GENERATOR/trt-llm/modeles/` ; à supprimer sur ordre, ou à garder comme copie du disque du comparatif.

## Reste
PPL TRT-LLM (classement ≤ 1,02× bf16) : instrument teacher-forcing à écrire (`return_context_logits`), verdict séparé. Puis GLM-4.7-Flash sur TRT-LLM (glm4_moe_lite : prise en charge à vérifier dans `_torch/models/`).
