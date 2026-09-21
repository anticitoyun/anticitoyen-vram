# § 1b (3e passe, .venv reconstruit avec cccl, 8a92ffbc) — 21/09

* instrument : `scratchpad/1b-precharger-21-09/chaine-parcbin.sh` (inchangé)
* commit : 8a92ffbc, worktree `poste2-8a92ffbc-21-09`, `.venv` reconstruit par `install.sh` (cuda-toolkit[nvcc,cccl] installé — corrige le `nv/target` manquant de la v2)
* régime : ACVRAM_ARBRE=$PWD
* scellé : 4/4 alias servent un jeton sous verrou, arrêt croisé propre → TENU
* mesuré :
  - acvram-coder-i8c : rc=0, 193 s, jeton décodé → **TENU** (le défaut `nv/target` de la v2 est résolu par le `.venv` reconstruit avec cccl)
  - llamacpp-gemma4-31b : rc=0, 164 s, jeton décodé → TENU
  - vllm-qwen3-vl-awq : rc=1, 157 s → toujours en défaut, même cause que les passes précédentes (FlashInfer absent du venv `/mnt/AI_GENERATOR/vLLM/venv`, hors mon domaine)
  - rapide-appoint : rc=0, 0 s, jeton décodé → TENU
* verdict : **RÉFUTÉ, 3/4** — progrès net sur les deux passes précédentes (2/4 → 3/4), seul vllm reste cassé. nvidia-smi avant/après identique (seul PID 4286, service permanent), arrêt croisé propre.
* durée : 517 s (13:16:38–13:25:16), prévu ≤ 8 min annoncé — dépassé (8:37), voir défaut de verrouillage ci-dessous.

## Défaut de verrouillage découvert pendant cette prise (important, hors scellé)
`parc/bin/acvram-serveur` et `parc/bin/llamacpp-serveur` (les lanceurs que la GUI utilise) allouent la carte directement (`setsid nohup ... &`) SANS jamais appeler `outils/carte.sh` — aucune ligne au journal du verrou pendant tout § 1b malgré un vrai chargement GPU en cours (confirmé par `nvidia-smi`/`ps` pendant la prise : llama-server PID 2589380 montant en VRAM, journal carte.sh muet). Signalé à la chef en direct (13:22) ; correctif confié à poste3 (prise de `carte.sh` en mode service, tenue pendant la vie du serveur).

## Suite
vllm-qwen3-vl-awq : `pip install flashinfer` dans `/mnt/AI_GENERATOR/vLLM/venv` (inchangé depuis la 1re passe).
