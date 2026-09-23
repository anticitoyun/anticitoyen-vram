# Protocole — refonte des PPL du comparatif sur le corpus privé (Sage § 8, main 1949ddb)

## Corpus (règle c)
`corpus-revue-5909d27.txt` = `acvram-memoire/revue/*.md` au commit **5909d27**, 283 fichiers triés par nom, concaténés par une ligne vide : 1 567 051 caractères. **sha256 : `76088761d41b2abc1b00eda66e0c37c91009c911fdf3dcc9b2137d5c0404f0a2`** (`scratchpad/corpus-prive/corpus-revue-5909d27.sha256`, refabriqué par `fabrique-corpus-prive.py`). Trois tranches par tokenizer (`tranches-corpus-16-09.py`, `PPL_CORPUS_SOURCE`) aux décalages 0 / 24 576 / 49 152 jetons, 24 577 jetons chacune, **retokenisation vérifiée identique** (Coder ×3, GLM ×3 : 24 577 = 24 577) — `scratchpad/corpus-prive/tranches-{coder,glm}/`. wiki-gptq devient la colonne « public » ; le seuil ≤ 1,02 se juge sur le privé seul ; un bras qui diverge de > 0,02 entre privé et public est signalé sans interprétation.

## Comment l'étalon bf16 a été mesuré (GLM comme Coder) — réponse à Sage
Instrument : `outils/glm-ppl-bf16-hf.py` (Manon, 15/09), copié en `scratchpad/glm-ppl-bf16-hf-16-09.py` / `coder-ppl-bf16-hf-16-09.py` avec trois variables d'environnement seulement (`PPL_CORPUS`, `PPL_MIN_CTX`, `PPL_MAX_TOK`) ; `transformers` `AutoModelForCausalLM` bf16, `device_map="auto"`, `max_memory={0: "26GiB", "cpu": "80GiB"}` (GLM 62 Go et Coder 57 Go délestés sur l'hôte par accelerate, ≈ 9 min de chargement, 12 fenêtres ≈ 6 min) ; logits fp32 ; fenêtres `ids[s:s+2048]` pour s = 0, 2048, … (les mêmes que `acvram eval`, `ppl-vllm-*`, `ppl-trtllm`, `llama-perplexity -c 2048`) ; cibles `chunk[first_new+1:]` avec `first_new = max(MIN_CONTEXT, …)` ; encodage `add_special_tokens=False` → **aucun BOS**.
BOS des autres bras : vLLM et TRT-LLM reçoivent des identifiants (aucun ajout) ; acvram `evaluate.perplexity` encode par `Tokenizer.encode` sans jetons spéciaux (docstring de Manon) ; llama.cpp : GGUF Coder `tokenizer.ggml.add_bos_token = false`, GGUF GLM sans la clé (`pre = glm4`, BPE → `add_bos` faux par défaut dans llama-vocab) — `llama-perplexity` n'insère donc de BOS dans aucun chunk. **Même instrument, mêmes fenêtres, même absence de BOS pour les deux modèles : l'étalon GLM n'est pas à refaire ; il est refait de toute façon sur le corpus privé.**

## Cadrage retenu : 12 × 1 024 cibles par tranche (min-ctx 1023), pas 12 × 2 047
`llama-perplexity` v1 note les positions 1024..2047 de chaque chunk ; son mode v2 (`--ppl-stride`) porte ici le chunk à 3 072 (`Calculation chunk = 3072` avec `-c 2048`, 11 chunks : journal local d'essai) et ne rend pas des fenêtres de 2 048. Tous les bras sont donc jugés à `PPL_MIN_CTX=1023` (cibles 1024..2047, 12 288 par tranche) — écart avec l'énoncé de Sage (2 047 cibles) dit ici, pas contourné.

## Bras et ordre (chaîne `scratchpad/ppl-prive-16-09/chaine.sh`, détachée)
bf16 Coder ×3 → bf16 GLM ×3 → par tranche : acvram Coder (MMA), acvram GLM `-k48` (`ACVRAM_MOE_MMA=0`), llama.cpp Coder Q4_K_M, llama.cpp GLM Q4_K_M (unsloth), vLLM Coder ModelOpt (KV auto), vLLM GLM GadflyII (KV fp8), TRT-LLM Coder ModelOpt (KV fp8 imposé). Chaque colonne du verdict portera : converti, calibration (« aucune » / « défaut collect.py » / « non publié » / « cnn_dailymail présumé »), corpus d'évaluation + sha256.

## Prédictions scellées (moyenne géométrique des 3 tranches, × bf16)
- acvram Coder : [1,010 ; 1,025] ; acvram GLM `-k48` : **≥ 1,000** (le 0,9956 de wiki-gptq ne se reproduit pas : 60 %) — **falsifié** si < 0,998 sur la moyenne.
- llama.cpp Coder : [1,010 ; 1,025] ; llama.cpp GLM : [1,030 ; 1,060].
- vLLM Coder ModelOpt : [1,08 ; 1,16] ; TRT-LLM Coder : [1,10 ; 1,20] ; vLLM GLM GadflyII : [1,03 ; 1,06].
- Divergence privé/public > 0,02 : attendue pour aucun bras (à signaler si elle survient).
