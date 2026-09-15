# Verdict — GLM-4.7-Flash sur TensorRT-LLM : refusé (deux blocages, dont un sans issue en amont)

instrument : contrôle de charge `LLM(model=GLM-4.7-Flash-NVFP4)` sous `carte.sh` (etat, 512 de contexte) — journal `scratchpad/trtllm-coder-16-09/glm-charge.log` ; lecture du code installé et du `main` GitHub
commit : arbre poste3 efe0571 ; TRT-LLM 1.3.0rc15
régime : converti `GadflyII/GLM-4.7-Flash-NVFP4` (compressed-tensors, NVFP4 g16, `Glm4MoeLiteForCausalLM` / `glm4_moe_lite`), le seul GLM NVFP4 présent ; pas de ModelOpt GLM
scellé : à la manière de P3 — charge OU refus explicite ; je prédisais le refus (architecture absente du registre, lu avant la prise)
mesuré : **refus** dès la lecture de la config : `ValueError: Unsupported quant_bits: 4. Supported: 8.` (`_torch/model_config.py:438-482` : compressed-tensors n'accepte que fp8 canal/bloc en rc15)
verdict : **TensorRT-LLM 1.3.0rc15 ne sert pas GLM-4.7-Flash, ni ce converti ni un autre : (1) compressed-tensors NVFP4 refusé — levé sur le `main` GitHub (branche `QuantAlgo.NVFP4`, g16) ; (2) `Glm4MoeLiteForCausalLM` absent du registre (`modeling_glm.py` n'enregistre que `Glm4MoeForCausalLM`, attention GQA, pas MLA) — encore absent sur `main`.** La colonne TRT-LLM de GLM dans la table (poste7 §2) se remplit « ne sert pas » ; un ModelOpt GLM ne changerait rien tant que (2) tient.

## Ce qui ouvrirait la colonne
- Un `modeling_glm_moe_lite` amont (MLA + MoE, comme DeepSeek-V3 dont TRT-LLM a le MLA) : à surveiller ; hors de portée d'un patch local raisonnable (MLA + chargeur de poids + noyaux).
- À ne pas faire : forcer le converti par `Glm4MoeForCausalLM` — architecture différente (MLA vs GQA), ce serait un chargement faux qui « marche ».

## Bilan TRT-LLM pour la table
    modèle       charge   b=1 t/s   b=12 t/s   b=12 J/j   prefill 2048   PPL (×bf16)
    Coder-30B    oui      234,9     2 104,7    0,177      55 419 j/s     10,617 (1,304 — non classé, converti ModelOpt 1,26× chez vLLM aussi)
    GLM-4.7      non      —         —          —          —              —
