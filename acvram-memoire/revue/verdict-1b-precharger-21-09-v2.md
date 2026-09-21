# § 1b (2e passe, parc/bin + ACVRAM_ARBRE) — 21/09

* instrument : `scratchpad/1b-precharger-21-09/chaine-parcbin.sh` (fichier suivi, ce commit)
* commit : 255a6a49 (worktree poste2, fusion c99f7e73 lanceurs parc/bin + ecaa69dd)
* régime : ACVRAM_ARBRE=$PWD (contourne le paquet installé, cf. verdict-glm-b1-ab-21-09.md), CUDA_VISIBLE_DEVICES=0
* scellé : 4/4 alias servent un jeton sous verrou, arrêt croisé propre → TENU
* mesuré :
  - acvram-coder-i8c : rc=1, 242 s — `source=arbre(poste2@255a6a49,propre)` confirmé (le contournement a pris), mais **nouveau défaut** : compilation JIT du noyau Marlin échoue — `acvram/kernels/marlin_port/libtorch_stable/quantization/marlin/marlin.cuh:6` inclut `cuda_fp16.h` du paquet pip `nvidia-cu13`, qui référence `#include <nv/target>` (CCCL/libcu++) absent de ce paquet minimal (`.venv/lib/python3.14/site-packages/nvidia/cu13/include/cuda_fp16.h:4492`) → `fatal error: nv/target: Aucun fichier ou dossier de ce nom`, `ninja: build stopped`. Différent du défaut de packaging de la 1re passe (bindings.cpp absent du paquet installé) : ici le fichier existe, c'est l'environnement de compilation (nvcc du wheel pip) qui manque un header transitif.
  - llamacpp-gemma4-31b (parc/bin) : rc=0, 186 s, jeton décodé → TENU
  - vllm-qwen3-vl-awq (parc/bin) : rc=1, 164 s — même défaut que la 1re passe (`serveur.log` : Engine core initialization failed, FlashInfer absent du venv vLLM)
  - rapide-appoint (parc/bin) : rc=0, 0 s, jeton décodé → TENU
* verdict : **RÉFUTÉ, 2/4** — mêmes deux alias tenus qu'à la 1re passe ; acvram change de cause d'échec (packaging → environnement de build nvcc), vllm identique (FlashInfer). nvidia-smi avant/après identique (seul PID 4286), aucune fuite VRAM.
* durée : 592 s (11:44:36–11:54:28), prévu ≤ 30 min, tenue

## Suite
1. acvram-coder-i8c : installer les headers CCCL/libcu++ manquants (`nv/target`) dans l'environnement de compilation — soit via `nvidia-cuda-cccl-cu13` (paquet pip complémentaire), soit en pointant `nvcc`/l'include path vers `/usr/local/cuda-13.4` (toolkit système complet) au lieu du wheel `nvidia-cu13` seul.
2. vllm-qwen3-vl-awq : inchangé, `pip install flashinfer` dans `/mnt/AI_GENERATOR/vLLM/venv`.
