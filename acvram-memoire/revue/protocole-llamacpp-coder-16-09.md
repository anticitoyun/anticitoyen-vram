# Protocole — llama.cpp officiel sur Coder-30B Q4_K_M (comparatif 5 moteurs)

Instrument : `scratchpad/banc-llamacpp-16-09.py` (decode : fenêtre ≥ 20 s + 7 passes courtes ; prefill : 7 rép. médian ; `energie.py` / `puissance_nvml`), `scratchpad/ppl-llamacpp-16-09.py` (llama-perplexity fenêtre par fenêtre, cadrage acvram eval reconstruit, contrôle de retokenisation) · commit : arbre laure à la prise · régime : binaire `/mnt/AI_GENERATOR/llamacpp/officiel` (4c9233c, CUDA 13.4, sm_120a), `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18 Go, 4,5 bpw), `-ngl 999 -np b -c 2304×b --no-cache-idle-slots -b/-ub 2048`, FA par défaut du binaire (auto), KV f16 ; une carte, plafond 400 W.

## Étalons (mesurés avant, même cadrage PPL : wiki-gptq 2048/2048/256, 4 fenêtres)
- bf16 Coder-30B par HF (`scratchpad/coder-ppl-bf16-hf-16-09.py`, copie du script de Manon) : **à mesurer** — l'étalon 8,1427 cité dans `verdict-ppl-trtllm-coder-16-09` est celui de GLM (erreur, erratum à suivre).
- acvram `Qwen3-Coder-30B-A3B-nvfp4` par `acvram eval --window 2048 --stride 2048 --min-context 256` : à mesurer (le 9,1218 du parc est à min-context 0).

## Prédictions (référence 14/09 `energie-brute-faible-lot-14-09`, binaire de Katy, lots de 2 000 jetons)
- D1 b=12 : t/s ∈ [675 ; 825] (747,8 ±10 %), J/jeton brut ∈ [0,40 ; 0,49] (Sage §4 : 0,444 ±5 %). **Falsifié** hors intervalles.
- D2 b=1 : t/s ∈ [290 ; 350] (319,9), J/jeton ∈ [1,05 ; 1,30]. **Falsifié** hors intervalles.
- P1 prefill 2048 : [2 500 ; 6 000] j/s. **Falsifié** hors.
- Q1 PPL Q4_K_M ≤ 1,02× bf16 → classé (70 %) ; **falsifié** > 1,02.
- I1 (instrument) : les 4 fenêtres retokenisent à 2 049 jetons exactement et llama compte 2 049 ; sinon la fenêtre est refusée et la PPL non publiée.

## Ce qui borne
Lots de 1 024 jetons/séquence (comme vLLM/TRT-LLM du comparatif) et non 2 000 comme le 14/09 : le prefill (256) pèse 20 % du lot ; la prédiction D1 en tient compte par l'intervalle ±10 %. Deux relevés de durée (`e.duree` NVML et perf_counter) ; `prompt_ms` interne du serveur publié pour le prefill à côté du chrono client.
