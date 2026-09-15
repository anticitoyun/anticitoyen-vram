# Protocole — TensorRT-LLM sur Coder-30B (comparatif 5 moteurs, P3 charge + décodage/prefill)

Instrument : `trtllm-bench throughput` (contrôle de charge, P3) puis `scratchpad/decode-trtllm-16-09.py` et `scratchpad/prefill-trtllm-16-09.py` (mêmes dénominateurs que les bras vLLM `decode-glm-vllm-16-09.py` / `prefill-glm-vllm-15-09.py`) · commit : arbre poste3 à la prise, TRT-LLM 1.3.0rc15 (`verdict-trt-llm-install-16-09.md`) · régime : converti `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (ModelOpt, `hf_quant_config` NVFP4), backend PyTorch, graphes CUDA, KV `auto` (dtype du checkpoint), `enable_block_reuse=False`, KV 0,85 · une carte (`CUDA_VISIBLE_DEVICES=0` sous `carte.sh`), 5090 seule pour NVML.

## P3 — contrôle de charge (poste7 §5.1)
`trtllm-bench --model <dossier> throughput --backend pytorch --dataset /mnt/AI_GENERATOR/trt-llm/charge-b12.jsonl --concurrency 12 --max_batch_size 12 --kv_cache_free_gpu_mem_fraction 0.85` (24 requêtes, 128 jetons de sortie).
- P3a : charge ET produit un rapport de débit → NVFP4 MoE servi sur sm_120 (CUTLASS, `fused_moe_cutlass.py:86-88`). **Falsifié** par un refus explicite nommant sm_120/NVFP4/MoE (résultat aussi, prédiction poste7 §4 seconde moitié tenue) ; un plantage sans message = échec d'instrument, pas de verdict.
- Je scelle P3a (charge), à cause de la déclaration lue dans le code ; probabilité subjective 70 %.

## Décodage (b = 1 et 12, fenêtre ≥ 20 s, invite 256, 1 024 jetons/séquence, ctx 2048)
Référence vLLM Coder-30B (poste3 acd91a7, 14/09, même fenêtre) : b=1 1,373 J/jeton ; b=12 1 437 t/s, 0,271-0,272 J/jeton ; acvram b=12 0,619 J.
- D1 : b=12 TRT-LLM entre 0,8× et 1,3× vLLM en t/s (1 150-1 870 t/s) et J/jeton 0,21-0,34 (poste7 §4 : 0,9-1,3× vLLM en J). **Falsifié** hors [0,8 ; 1,3]×.
- D2 : b=1 TRT-LLM ≥ 0,9× vLLM b=1 en t/s (à lire dans acd91a7). **Falsifié** < 0,9×.
- 7 passes : la fenêtre 20 s est UNE passe longue (J/jeton, protocole §3) ; pour t/s, 7 lots courts de 128 jetons (`BANC_JETONS=128`, `BANC_FENETRE_S=0` = 1 lot) sont relevés en plus, meilleure passe retenue — à ajouter au script si `BANC_FENETRE_S=0` n'est pas géré (il ne l'est pas : premier lot exécuté quoi qu'il arrive).

## Prefill (L = 2048, 7 répétitions, médian)
Référence : vLLM Coder-30B — pas de chiffre apparié dans mes verdicts (GLM : 26 732 j/s). Je scelle P-1 : TRT-LLM prefill 2048 ≥ 15 000 j/s ; **falsifié** < 15 000 (ordre de grandeur, pas un classement).

## Ce qui ne se conclut pas ici
PPL TRT-LLM (≤ 1,02× bf16 pour être classé) : instrument à écrire (logprobs teacher-forcing via `LLM.generate(..., return_context_logits)`) — après le chiffre de temps, verdict séparé.
