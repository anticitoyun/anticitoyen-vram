# Verdict — installation TensorRT-LLM (comparatif 5 moteurs, phase 1)

instrument : uv 0.12.13, venv `/mnt/AI_GENERATOR/trt-llm/.venv` (python 3.12), `import tensorrt_llm` sous `carte.sh` (etat)
commit : ccdf678 (main d529cc9), protocole `protocole-trt-llm-install-16-09.md` (16809ed)
régime : à sec ; un seul passage GPU de 20 s pour l'import (verrou `etat`, carte vide avant/après)
scellé : P1 résolution uv sans conflit · P2 `trtllm-bench` présent · venv 8-12 Go · P3 (charge Coder-30B ModelOpt) en attente
mesuré : P1 **réfutée** puis corrigée · P2 tenue · venv 15 Go (falsifié > 20 : non) · P3 **non exécutée** (converti absent)
verdict : **TensorRT-LLM 1.3.0rc15 installé et importable sur la RTX 5090 (sm_120, torch 2.10.0+cu130, CUDA 13.0)** ; le contrôle de charge attend le téléchargement du converti ModelOpt.

## Ce qui a été fait
- `install-1-conflit.log` : le wheel NVIDIA rc15 exige `torch>=2.10.0,<=2.11.0a0` (le protocole disait 2.12/2.13 : lu sur une autre source, faux) ; le torch 2.10.0 de pypi.org est cu128 (`cuda-bindings==12.9.4`) et TRT-LLM exige `cuda-python>=13` → insoluble. Correction : `torch==2.10.0+cu130` depuis `download.pytorch.org/whl/cu130`, `--index-strategy unsafe-best-match`.
- `install-2-tue.log` : uv tué deux fois par le superviseur de session (« low memory », 75 Go disponibles pourtant) ; relancé en unité `systemd-run --user trtllm-install` → `install.log`, rc=0.
- `import tensorrt_llm` échouait `libmpi.so.40` absent (pas d'OpenMPI système, sudo requis) → `brew install open-mpi` 5.0.10, `LD_LIBRARY_PATH` dans `/mnt/AI_GENERATOR/trt-llm/env.sh` (à sourcer).
- Import sous verrou : `1.3.0rc15 2.10.0+cu130 13.0 (12, 0)`. Sans carte visible l'import lève `No CUDA GPUs are available` (`cuda_tile_utils.py:53` interroge le GPU à l'import) : `trtllm-bench --help` lui-même exige le verrou.
- Pile : tensorrt 10.15.1.29 (cu13), cuda-python 13.0.3, nvidia-nccl-cu13 2.28.9, flashinfer 0.6.11.post1, nvidia-modelopt 0.37.0, cutlass-dsl 4.5.0, triton 3.6.0.

## Ce que dit le code sur sm_120 (à sec, non mesuré)
- `_torch/modules/fused_moe/fused_moe_cutlass.py:86-88` : NVFP4 déclaré pour SM ∈ {100, 103, 120, 121} (chemin CUTLASS).
- `fused_moe_trtllm_gen.py:228-230` : le backend TRTLLM-Gen refuse SM ≥ 120.
- Donc la prédiction Sage §4 « ne sert pas NVFP4 MoE sur sm_120 » est contredite par la déclaration, pas encore par une charge : seul P3 tranche.

## Reste (P3, sous verrou)
- Converti `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (ModelOpt, 15,6 G params, ≈17 Go) : le cache HF n'a que `refs/main`, aucun snapshot — téléchargement à faire (engage l'utilisateur, §5.2).
- Commande prévue : `source env.sh && trtllm-bench --model <chemin> throughput --backend pytorch ...` ; résultat écrit quel qu'il soit (charge, refus nommant sm_120/NVFP4/MoE, ou plantage = échec d'instrument).
