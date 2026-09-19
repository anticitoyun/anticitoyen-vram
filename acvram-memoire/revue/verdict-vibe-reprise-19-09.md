instrument · Vibe
commit · 9099894
régime · CUDA_VISIBLE_DEVICES=""
scellé · REPRISE.md §2-10 relu
mesuré · acvram/__init__.py:19 revue/comparatif-cinq-moteurs-17-09.md:15,32
verdict · texte de remplacement proposé pour §2 et §10

## §2 État actuel (remplacement proposé)
Version 0.6.13 (acvram/__init__.py:19). 81 tests. Sur machine cible : noyaux compilés sm_120+sm_86, graphes CUDA au décodage, sources GGUF. Qwen3-Coder-30B-A3B : b=1 363,3 t/s, b=12 1 361,4 t/s, PPL 1,0155 géo (revue/comparatif-cinq-moteurs-17-09.md:15). Développé sur portable i5-3230M/GT 740M/pilote 470 : aucun code CUDA jamais compilé ni exécuté.

## §10 Ce qui vient ensuite (remplacement proposé)
1. Noyau CUDA d'attention paginée
2. Cache LRU d'experts fréquents (modélisé, non implémenté)
3. Prefill par morceaux (machinerie existe, ordonnanceur à découper)
4. Spéculation à la EAGLE
5. GEMM groupé pour les MoE
6. Compensation d'erreur à la GPTQ
Pistes non engagées : split-K b=1 opt-in ACVRAM_GEMV_SPLITK (revue/verdict-splitk-b1-19-09).
