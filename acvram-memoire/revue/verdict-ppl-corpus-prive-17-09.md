# Verdict — PPL du comparatif sur le corpus privé : Coder jugé (llama.cpp seul classé), GLM injugeable (référence bf16 instable)

instrument : chaîne `scratchpad/ppl-prive-16-09/chaine.sh` + `chaine2.sh` (vLLM/TRT-LLM rejoués : premier passage tombé sur un chemin de sortie relatif après `cd`) ; bf16 HF (`*-ppl-bf16-hf-16-09.py`), acvram `reppl-eval-16-09.py`, `ppl-llamacpp-16-09.py`, `ppl-vllm-{coder,glm}-16-09.py`, `ppl-trtllm-16-09.py` ; seconde référence GLM : vLLM bf16 délesté (`cpu_offload_gb=42`, KV bf16) — journaux `scratchpad/ppl-prive-16-09/`
commit : arbre poste3 ec199c5 ; protocole `protocole-ppl-corpus-prive-17-09.md`
régime : corpus privé `corpus-revue-5909d27.txt`, sha256 `76088761d41b2abc1b00eda66e0c37c91009c911fdf3dcc9b2137d5c0404f0a2`, 3 tranches × 12 fenêtres de 2 048, cibles 1024..2047 (12 288 par tranche) ; calibrations : acvram Coder « aucune » (awq=False), acvram GLM « défaut collect.py ou --calib-file non écrit », GGUF unsloth « imatrix non publié », ModelOpt communautaire « non publié (cnn_dailymail présumé) »
scellé : acvram Coder [1,010 ; 1,025] · acvram GLM ≥ 1,000 (60 %) · llama.cpp Coder [1,010 ; 1,025] · llama.cpp GLM [1,030 ; 1,060] · vLLM Coder [1,08 ; 1,16] · TRT-LLM Coder [1,10 ; 1,20] · vLLM GLM [1,03 ; 1,06]
mesuré (moyenne géométrique des 3 tranches × bf16) : Coder — **llama.cpp 1,0126** (tenue) · **acvram 1,0271** (**réfutée**, > 1,025 ; tranche 2 : 1,036) · vLLM ModelOpt 1,1558 (tenue) · TRT-LLM 1,2340 (**réfutée**, > 1,20). GLM — bf16 HF 32,59 / 26,62 / 15,87, seconde référence vLLM bf16 tranche 1 = **27,92** (HF 26,62 : les deux bf16 concordent à 5 %) ; acvram 1,030 / **0,781** / 0,907, llama.cpp 1,029 / 0,932 / 0,865, vLLM NVFP4 1,092 / **0,569** / 0,722 — trois convertis 7 à 43 % **sous** bf16 sur deux tranches ; scellés GLM ni tenus ni réfutés : la référence ne tient pas.
verdict : **Coder-30B, corpus privé : seul llama.cpp Q4_K_M est classé (1,013×) ; acvram nvfp4 sort à 1,027 (non classé, il l'était de peu sur wiki-gptq : divergence privé/public 0,009, sous 0,02) ; vLLM/TRT-LLM ModelOpt loin (1,16 / 1,23). GLM-4.7-Flash : injugeable sur ce corpus — la PPL bf16 elle-même explose par fenêtres (llama.cpp : fenêtres à 107 090, 8 402, 262 ; HF bf16 tranche 0 = 32,6 contre des fenêtres ordinaires à 9-25), et les convertis passent sous la source de 10 à 43 % ; ce n'est pas un classement, c'est une instabilité numérique du modèle sur ce texte, présente dans chaque moteur à des fenêtres différentes.**

## Tableau (× bf16, moyenne géométrique des 3 tranches ; « public » = wiki-gptq du 16/09 au même cadrage)
    modèle   bras / converti                 calibration              privé     public    diverg.   classé (privé)
    Coder    llama.cpp Q4_K_M (unsloth)      imatrix non publié       1,0126    1,0146    0,002     oui
    Coder    acvram nvfp4                    aucune (awq=False)       1,0271    1,0180    0,009     non
    Coder    vLLM ModelOpt NVFP4/…-FP4       non publié               1,1558    1,123*    0,033     non — divergence > 0,02 signalée
    Coder    TRT-LLM même ModelOpt (KV fp8)  non publié               1,2340    1,164*    0,070     non — divergence > 0,02 signalée
    GLM      acvram -k48 / llama.cpp / vLLM  (voir régime)            0,900 / 0,940 / 0,766   0,996 / 1,043 / 1,043*   —   injugeable
    * public sur 4 fenêtres (4 096 notés), pas 3 tranches. bf16 privé : Coder 11,73 (12,46 / 11,27 / 11,51), GLM 23,97 (32,59 / 26,62 / 15,87) ; bf16 public : Coder 7,99, GLM 7,74.

## Ce que dit la fenêtre par fenêtre (llama.cpp, seul instrument à la publier)
Coder : 12 fenêtres par tranche entre 6,1 et 23,2 — régulier. GLM : tranche 0 [9,2 … 24,9, **107 090**, 16,8, 7,2, 22,5], tranche 1 [… **8 402**, 22,2], tranche 2 [… **262**, …] ; les secondes moitiés de ces fenêtres sont de la prose ordinaire (extraits dans le journal). Le corpus ne contient aucun jeton spécial GLM (vérifié : 0 sur 3 × 24 577). La divergence entre moteurs sur GLM vient de ces fenêtres et de leur traitement numérique différent (KV int8 / f16 / fp8, accumulations), pas du texte.

## Ce qui manque avant de juger GLM (à poste7)
1. PPL **par fenêtre** dans chaque instrument (HF bf16, acvram, vLLM, TRT-LLM : à ajouter — llama.cpp l'a) et un agrégat robuste écrit d'avance (médiane des fenêtres, ou exclusion des fenêtres où bf16 > 100 avec leur nombre publié).
2. Comprendre l'explosion bf16 de GLM sur ce texte (softmax MLA en bf16 ? position ? ligne fautive) — une fenêtre, des logits lus (leçon `precision-de-sortie-invisible-a-la-ppl`). Tant que ce n'est pas fait, aucune PPL GLM n'est un classement.
3. Coder ne montre rien de tel : la table Coder est publiable telle quelle.
