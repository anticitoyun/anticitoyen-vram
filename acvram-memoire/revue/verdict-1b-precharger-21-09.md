# § 1b — commande exacte de precharger(), 4 moteurs (21/09)

* instrument : `scratchpad/1b-precharger-21-09/chaine.sh` + `verifier.py` (fichiers suivis, ce commit)
* commit : 48f9ecb5 (worktree poste2 fusionné sur main avant la prise)
* régime : ACVRAM_CARTE=0, aucune variable de régime posée (précharge de service, pas une mesure d'énergie)
* scellé : 4/4 alias servent un jeton sous verrou, arrêt croisé propre → TENU
* mesuré :
  - acvram-coder-i8c (acvram-serveur, coder-30b-a3b-qkvo-i8c) : rc=1, 244 s — `/tmp/acvram-serveur.log` : `introuvable : .../acvram/kernels/marlin_port/bindings.cpp` (source manquante dans le venv installé, pas d'OOM sur cette prise)
  - llamacpp-gemma4-31b (65536 ctx) : rc=0, 272 s, jeton décodé, modele_servi = alias attendu → TENU
  - vllm-qwen3-vl-awq (131072 ctx) : rc=1, 245 s — `/mnt/AI_GENERATOR/vLLM/serveur.log:1563` : `RuntimeError: FlashInfer backend is not available` (dépendance absente du venv vLLM, têtes non multiples de 32 sans repli)
  - rapide-appoint (llamacpp-appoint, service permanent) : rc=0, 0 s, jeton décodé → TENU
* verdict : **RÉFUTÉ, 2/4** (llamacpp et rapide-appoint tenus ; acvram et vllm cassent chacun sur un défaut d'environnement nommé, aucun rapport entre eux) — nvidia-smi avant/après identique (seul PID 4286, service 8081), aucun process fantôme, aucune fuite VRAM
* durée : 1002 s (10:04:21–10:17:03), prévu ≤ 30 min, tenue

## Suite
1. acvram-coder-i8c : réinstaller/reconstruire le paquet de noyaux acvram dans `~/.local/share/acvram/venv` (bindings Marlin manquants) — hors mesure, geste d'installation.
2. vllm-qwen3-vl-awq : `pip install flashinfer` (ou repli explicite) dans `/mnt/AI_GENERATOR/vLLM/venv` avant de reprendre ce bras.
