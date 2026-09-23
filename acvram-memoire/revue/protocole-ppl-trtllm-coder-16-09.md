# Protocole — PPL TensorRT-LLM, Coder-30B NVFP4 ModelOpt (comparatif, critère ≤ 1,02× bf16)

Instrument : `scratchpad/ppl-trtllm-16-09.py` — logits de contexte bruts (`return_context_logits=True`), alignés logits[t] → f[t+1], cadrage `acvram eval` / `ppl-vllm-glm-16-09.py` (wiki-gptq, 2048/2048/256, 8192 jetons, 4 fenêtres × 1791 = 7164 notés) · commit : arbre laure (ce commit) · régime : même converti que le verdict de temps (KV **fp8** du checkpoint, `BANC_KV=auto`), backend PyTorch, b=1 · étalon bf16 : **8,1427** (`verdict-ppl-decode-mma-coder30b-15-09`, même cadrage).

## Prédictions
- P1 : PPL TRT-LLM ∈ [8,20 ; 8,40]. **Falsifié** hors de l'intervalle.
- P2 (critère du comparatif) : ≤ 1,02 × 8,1427 = **8,306** → classé. Je scelle « classé » à 60 % (les NVFP4 ModelOpt de Coder chez vLLM/acvram tiennent sous 1,01 ; le KV fp8 ajoute ≈ 0,3-0,5 %).
- P3 (bras qui doit différer) : l'alignement logits[t] ↔ f[t] de `compute_logprobs` (`executor/result.py:1027-1031`) rend une PPL > 100. **Falsifié** s'il rend une PPL proche de P1 : alors les `context_logits` de TRT-LLM sont déjà décalés et c'est MON alignement qui serait faux — l'instrument ne publie rien tant que P3 n'est pas tranchée.
- Sanité : `logits_dtype` publié ; si bf16, noter que la PPL est celle d'un lm_head bf16 (leçon `precision-de-sortie-invisible-a-la-ppl`).

## Ce qui ne se conclut pas
Le KV fp8 est imposé par le checkpoint ; une PPL en KV bf16 (`BANC_KV=bf16`, si le moteur l'accepte contre `hf_quant_config`) sera publiée si P2 tombe entre 8,306 et 8,40, pour attribuer l'écart.
