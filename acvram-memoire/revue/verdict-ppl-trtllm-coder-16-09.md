# ERRATUM (16/09, ~19:00, Laure) — l'étalon 8,1427 est celui de GLM, pas de Coder
Le rapport ×1,304 ci-dessous compare à **8,1427 = PPL bf16 de GLM-4.7-Flash** (`verdict-glm-ppl-finale-15-09`), cité par erreur : il n'existait pas d'étalon bf16 Coder-30B dans le dépôt. Mesuré depuis, même cadrage (HF bf16 `device_map=auto`, `scratchpad/coder-ppl-bf16-hf-16-09.py`, 7 164 notés) : **bf16 Coder-30B = 9,1747** ; acvram `Qwen3-Coder-30B-A3B-nvfp4` (`reppl-eval-16-09.py`, MMA, piles 48/48) = **9,2833**. Rapports corrigés : TRT-LLM 10,617 → **1,157×** (non classé, > 1,02) ; vLLM mêmes poids 10,262 → **1,119×** (non classé) ; acvram nvfp4 → **1,012×** (classé). Les chiffres mesurés et l'attribution (converti 1,12×, moteur +3,3 %) tiennent ; seuls les rapports changent. Journaux : `scratchpad/llamacpp-coder-16-09/ppl-bf16-hf.log`, `ppl-acvram-nvfp4.json`.

# Verdict — PPL TensorRT-LLM, Coder-30B NVFP4 ModelOpt : non classé (1,304× bf16), le checkpoint porte 1,26×

instrument : `scratchpad/ppl-trtllm-16-09.py` (logits de contexte bruts, alignés logits[t]→f[t+1], log-softmax fp32 sur CPU) ; témoins `scratchpad/ppl-vllm-coder-16-09.py` (vLLM, `prompt_logprobs=0`, TRITON_ATTN) — journaux `scratchpad/trtllm-coder-16-09/ppl-*.{log,json}`
commit : 2ecd4d4 + correctifs KV 0,20 / shutdown (commit de ce verdict) ; protocole `protocole-ppl-trtllm-coder-16-09.md` (c71ce31)
régime : cadrage `acvram eval` (wiki-gptq, 2048/2048/256, 4 fenêtres, 7 164 jetons notés), même converti `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` pour les trois bras ; TRT-LLM KV **fp8 imposé** (`bfloat16` refusé : « Accepted types are fp8, nvfp4, auto »), logits fp32 ; vLLM KV auto (bf16) et fp8
scellé : P1 PPL ∈ [8,20 ; 8,40] · P2 ≤ 8,306 classé (60 %) · P3 alignement t↔t aberrant (> 100)
mesuré : P1 **réfutée** : **10,617** · P2 **réfutée** : 1,304× bf16 (8,1427), non classé · P3 tenue : 223 751 (l'alignement de `compute_logprobs` est bien faux d'un jeton ; mon instrument voit le décalage) · témoins vLLM même checkpoint : **10,262** (KV bf16), **10,278** (KV fp8)
verdict : **TensorRT-LLM n'est pas classable sur Coder-30B avec ce converti : 1,304× bf16. L'écart vient d'abord du checkpoint ModelOpt communautaire (vLLM sur les mêmes poids : 1,260×), le moteur ajoute +3,3 % à KV égal (10,617 / 10,278) — pas le KV fp8 (vLLM : +0,16 %).** Le débit ×1,46 vLLM du verdict de temps est donc mesuré sur un converti qui ne passe pas le critère du comparatif ; il reste publié avec cette réserve.

## Ce que ça change pour la table (Sage §2)
- La colonne « NVFP4 ModelOpt (à télécharger) » pour vLLM et TRT-LLM pointe sur un converti à 1,26× : ni l'un ni l'autre ne se classe avec lui. Il faut un ModelOpt refait depuis `srcbf16` (calibration, `exclude_modules` : le checkpoint communautaire quantifie attention + experts, `kv_cache_quant_algo: FP8`) — conversion à sec chez Manon (`modelopt` 0.37.0 est dans le venv TRT-LLM), ou une source ModelOpt officielle NVIDIA si elle existe pour ce modèle.
- Le +3,3 % propre à TRT-LLM (même poids, même KV) est un fait du moteur (noyaux MoE CUTLASS NVFP4 / attention TRTLLM sur sm_120) ; il se retesterait sur un converti sain avant d'être attribué.

## Bornes
- Une seule fenêtre de 8 192 jetons (bruit 0,008 en PPL, `sage-narrow-verdict` § 6) : les écarts ici (×1,26, +3,3 %) sont 30 à 300 fois le bruit.
- Le bras vLLM utilise les `prompt_logprobs` (alignés par vLLM) ; le bras TRT-LLM mes propres logits : deux instruments, même cadrage, même tokenizer (7 164 notés des deux côtés).
- Pièges d'instrument consignés : `prompt_logprobs` TRT-LLM décalé d'un jeton (`executor/result.py:1027-1031`) ; les `context_logits` arrivent sur le GPU du processus principal (1,16 Gio par fenêtre) pendant que le worker tient `free_gpu_memory_fraction` → OOM à 0,80 et 0,60, passe à 0,20 ; une exception sans `shutdown()` laisse le worker MPI sur la carte (verrou tenu) → `finally`.

## Reste
GLM-4.7-Flash sur TRT-LLM (prise en charge `glm4_moe_lite` à vérifier), puis la question du converti ModelOpt sain pour Coder — à Sage via Jérôme.
