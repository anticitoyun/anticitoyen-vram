# Verdict — cellule Coder-30B × EXL3 (exllamav3, moteur de TabbyAPI) vitesses : b=12 856 t/s / 0,362 J, b=1 168 t/s, prefill 10 656 j/s

instrument : `scratchpad/decode-exl3-17-09.py` (Generator/Job exllamav3, ArgmaxSampler, aucun arrêt EOS, cache fp16 ; décodage : fenêtre ≥ 20 s en lots de 1 024 jetons/séquence + 7 passes courtes de 128, `energie.py` ; prefill : Job max_new_tokens=1, L=2048, 2 chauffes, 7 rép., `puissance_nvml`) — journaux `scratchpad/exl3-vitesses-17-09/`
commit : arbre laure cecfb18 ; converti `Qwen3-Coder-30B-A3B-Instruct-EXL3-4.25bpw` (Manon ; bpw mesuré 4,283 / tête 6,008) ; venv `/opt/ia/TabbyAPI/.venv`, dépôt exllamav3 `/tmp/exllamav3-repo`
régime : moteur exllamav3 en direct (pas le serveur HTTP TabbyAPI : même générateur, sans la couche HTTP), `ACVRAM_DISABLE_KERNELS=1` posé pour empêcher `energie.py` (qui importe `acvram`) de compiler l'extension dans ce venv — une première passe l'avait fait pendant la fenêtre b=1 (134,7 t/s, jetée : `decode-v1-import-acvram.log`) ; une carte, invites de 256 jetons tirés (mêmes que vLLM/TRT-LLM)
scellé (Sage) : b=12 ≤ 0,6 × TRT-LLM (≤ 1 263 t/s) · b=1 ≥ 250 t/s
mesuré : **b=1 : 168,0 t/s, 1,546 J brut (1,096 net), 260 W** · **b=12 : 855,7 t/s, 0,362 J brut (0,272 net), 310 W, aucune invalidation** (meilleure passe courte 661,6) · **prefill pp2048 : 10 656 j/s** (σ 140, 192 ms)
verdict : **b=12 tenu (856 = 0,41 × TRT-LLM), b=1 réfuté (168 < 250 : exllamav3 est le plus lent des cinq à b=1, sous acvram 234 et llama.cpp 341). À b=12 il est devant acvram (730) et llama.cpp (709), derrière vLLM (1 438) et TRT-LLM (2 105), avec la meilleure PPL de la table (1,0006) et 0,362 J/jeton (acvram 0,546). Prefill 10 656 j/s : entre acvram (8 633) et llama.cpp (15 717).**

## Bornes
- Pas de bridage puissance (260-310 W) : la carte n'est pas saturée par ce moteur ; le b=1 à 168 t/s est un coût de lancement (générateur Python, pas de graphes CUDA côté exllamav3 par défaut), pas de bande passante.
- Meilleure passe courte b=12 (662) < fenêtre (856) : les lots de 128 jetons payent le prefill de 12 × 256 ; la fenêtre est le chiffre du régime.
- Le serveur TabbyAPI ajouterait la couche HTTP/JSON ; non mesuré (le comparatif compare les moteurs, comme pour vLLM et TRT-LLM en API Python).
