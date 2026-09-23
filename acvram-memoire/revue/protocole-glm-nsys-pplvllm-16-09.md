# Protocole — après duel : nsys d'un pas b=12 GLM -k48 (en-tête du chantier MLA) et PPL vLLM même corpus

Laure, 16/09/2026, avant mesure. Ordre : Jérôme (main 3d96697, chantier MLA ouvert), 15 min, pas de ncu.
Arbre : travail/laure au commit de ce protocole (code `acvram/` = main ce71723).

## Montage
1. **nsys** `--cuda-graph-trace=node -t cuda,nvtx`, `nsys-rejeu-b12-15-09.py` (50 pas de décodage pur
   b=12, ctx 2048, invite 256, synchronize par pas, **régime du duel** : `ACVRAM_MOE_MMA=0`,
   décodage MMA=1 MIN_T=5), `-k48` ; analyse `nsys-trous-analyse-15-09.py` + poste **mla** (regex
   `mla|absorb|latent|kv_lora|decode_attention|_attn_fwd|paged_attn`) placé avant `attention`,
   poste `denses` (gemv/int8_gemm/narrow). Sortie : ms/pas par poste, trou = pas − union.
2. **PPL vLLM** `ppl-vllm-glm-16-09.py` : même corpus, mêmes 4 fenêtres de 2048 (stride 2048),
   mêmes cibles `chunk[257:2048]` (1 791 × 4 = 7 164 notées, comme HF), tokenizer du modèle
   GadflyII `add_special_tokens=False`, `prompt_logprobs=0`, **KV fp8 + patch MLA** = régime du duel.
   Référence bf16 8,1427.

## Mes prédictions (scellées)
1. Pas b=12 sous nsys **58-64 ms** (56,1 en rondes + synchronize). Postes : **mla 28-36 ms
   (50-60 %)**, moe_gemm + quant + glue **10-14 ms**, denses (int8, 2,0 Go) **2-4 ms**, trou
   **1-3 ms** (le 0,78 ms de Coder avec 1 517 lancements ; GLM en a plus : 12 `torch.cat` par
   couche au décodage MLA, `model.py:1252` — si le trou dépasse 5 ms, c'est le lancement, pas
   les noyaux). Réfuté si mla < 40 % du pas : alors le poste est ailleurs (denses ou trous)
   et le chantier MLA n'est pas le premier.
2. PPL vLLM GadflyII : **1,000-1,006** (NVFP4 experts + denses, KV fp8 ; recette inconnue
   d'AWQ). Si > 1,010, le duel compare des qualités différentes et la colonne le dit ;
   si < 0,998, GadflyII a un avantage de recette à nommer.
Unité `glm-nsys-pplvllm-laure`, sorties `scratchpad/glm-nsys-pplvllm-16-09/`.
