# Protocole — installation TensorRT-LLM (comparatif 5 moteurs, phase 1)

Instrument : `uv pip install` dans `/mnt/AI_GENERATOR/trt-llm/.venv` (python 3.12) · commit : ccdf678 (main d529cc9 fusionné) · régime : à sec, hors verrou carte, aucune mesure GPU · date : 16/09/2026.

## Faits d'environnement (lus avant de choisir)
- CUDA 13.4 (`/usr/local/cuda-13.4`), pilote 595.91.07, RTX 5090 (sm_120), python système 3.14.5, `/mnt/AI_GENERATOR` 1,4 T libre.
- pypi.org `tensorrt-llm` : sdist seulement (stables ≤ 1.2.1, rc ≤ 1.3.0rc26) — compilation source exclue (heures, bazar CUDA).
- Index NVIDIA `https://pypi.nvidia.com/tensorrt-llm/` : wheels cp312 linux pour 1.1.0, 1.2.0, 1.2.1 (2,5 Go) et jusqu'à 1.3.0rc15.
- 1.2.1 exige `torch<=2.10.0a0,>=2.9.1` + `tensorrt~=10.14.1` ; 1.3.0rc15 exige `torch<=2.13.0a0,>=2.12.0a0`, `cuda-python>=13`, `nvidia-nccl-cu13`, `nvidia-cutlass-dsl[cu13]==4.6.2`.
- torch 2.12.1 sur pypi.org est cu13 par défaut (dépend de `nvidia-cudnn-cu13==9.20.0.48`, `nvidia-nccl-cu13==2.29.7`) : pas d'index PyTorch spécial nécessaire.

## Choix scellé
**1.3.0rc15** (dernier binaire cp312) : la seule voie binaire alignée sur CUDA 13 et la plus récente pour sm_120 ; la stable 1.2.1 impose torch 2.9/2.10 et TensorRT 10.14, un cran plus vieux pour Blackwell grand public. Repli si rc15 refuse : 1.2.1 sur le même venv recréé.

## Prédictions (avant de lancer)
- P1 : la résolution uv aboutit sans conflit et `python -c "import tensorrt_llm"` passe. **Falsifié** si uv rend un conflit de versions ou si l'import lève (ABI, libnccl, cuda-python).
- P2 : `trtllm-bench --help` s'exécute dans le venv. **Falsifié** si le point d'entrée manque.
- P3 (contrôle de charge, sous verrou, plus tard) : `trtllm-bench` sur Coder-30B ModelOpt NVFP4 charge OU refuse avec un message nommant sm_120/NVFP4/MoE — les deux sont des résultats ; ce qui serait un échec d'instrument : un plantage sans message (segfault, OOM à vide).
- Taille : venv ≈ 8-12 Go (wheel 2,5 Go + torch 0,5 Go + libs CUDA). **Falsifié** si > 20 Go.

## Commandes
```
uv venv --python 3.12 /mnt/AI_GENERATOR/trt-llm/.venv
uv pip install --python /mnt/AI_GENERATOR/trt-llm/.venv/bin/python \
  --extra-index-url https://pypi.nvidia.com "tensorrt_llm==1.3.0rc15" \
  2>&1 | tee /mnt/AI_GENERATOR/trt-llm/install.log
```
Aucun GPU touché ; le contrôle de charge (P3) attend le converti ModelOpt (téléchargement ~20 Go, engage l'utilisateur) et le verrou.
