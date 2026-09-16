# Verdict — trois déquantifications d'un expert NVFP4 (ordre poste7, 17/09) : vLLM = nvfp4.py au bit ; noyau à mesurer sur carte ; le candidat « ordre bloc × global » ne peut pas produire d'écart

- **régime** : à sec (CPU), tenseurs lus tels quels dans les sources de vLLM ; script
  `outils/dequant-trois-references-17-09.py` (chemin torch de `dequantize_to_dtype`,
  `swizzle=False`, `is_cuda_alike` forcé faux ; notre `dequantize_nvfp4` sur le `NVFP4Tensor`
  du passage direct ; le noyau `nvfp4_dequant` quand une carte est là).
- **scellé** : max |Δ| ≤ 1 ulp bf16 entre les trois. Prédiction de poste7 : vLLM et nvfp4.py s'écartent
  (ordre `block_scale × global` fp32 → bf16).
- **mesuré** (3 experts par source, `layers.3.mlp.experts.7-9.gate_proj`) :
  - GLM-4.7-Flash-NVFP4 (compressed-tensors, global = 1/weight_global_scale) : vLLM vs nvfp4.py
    **0,000 ulp, identiques au bit** (3/3) ; échelle globale effective **1,0** sur cette source.
  - Qwen3-Coder-30B-A3B-Instruct-FP4 (modelopt, weight_scale_2 ≈ 5e-5 … 1e-4) : vLLM vs nvfp4.py
    **0,000 ulp, identiques au bit** (3/3).
  - noyau `nvfp4_dequant` : **non mesuré** (pas de carte) — même script, 1 min, poste3.
- **verdict** : **TENU** pour les deux références ; la prédiction de poste7 est **réfutée par
  construction**, pas seulement par la mesure : un code E2M1 (≤ 3 bits significatifs) × une
  échelle E4M3 (4 bits) est un produit à ≤ 8 bits significatifs, exactement représentable en
  bf16 ; arrondir avant ou après le produit par l'échelle globale ne change rien (le témoin
  « global après arrondi bf16 » rend 0 ulp aussi, sur les deux sources). Sur GLM, l'échelle
  globale vaut 1 : la question ne se pose même pas.
- **conséquence** : si le 1 % de PPL entre acvram et vLLM sur -vllm-direct survit au scellé 2, il
  n'est pas dans la déquantification des poids ; restent le noyau (à mesurer, ci-dessus), les
  activations (vLLM : échelle globale STATIQUE `input_global_scale` par tenseur ; nous :
  dynamique par ligne — en W4A16 sans objet), la tête/normes en bf16 chez eux (honorées depuis
  8164622), et l'instrument PPL lui-même (préfixe, fenêtres).
- **reste** : bras noyau sur carte ; re-PPL -vllm-direct reconverti avec 8164622.
