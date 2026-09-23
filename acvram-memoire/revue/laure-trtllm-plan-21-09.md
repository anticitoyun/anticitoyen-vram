# Plan — Installation TensorRT-LLM pour sm_120 (21/09, Laure, à sec)

**Versions disponibles :** v1.3.0rc22, v1.3.0rc23 (docs source) ; releases stables antérieures sur GitHub/NVIDIA.

**Cible architecture :** sm_120 (Ada : RTX 5090, L40, H200) — supporté nativement, skip_softmax FMHA hand-written warp-specialized pour ce GPU.

## Prérequis système

| élément | requis | notes |
|---|---|---|
| CUDA Toolkit | 13.2 exact | `CUDA_HOME` doit être défini |
| Driver NVIDIA | >= 13.2 compat | `cuda-compat-13-2` peut être nécessaire selon le driver |
| Python | 3.10+ | confirmé dans pyproject.toml |
| CMake | >= 3.27 (ou 3.31 pour FP4) | si build from source |
| Compilateur | clang, lld, llvm | requis pour C++ runtime |

## Option 1 : Virtual Environment (pip)

**Taille estimée :** 3–4 Go (roues + dépendances PyTorch 2.13.0) + 2–3 Go pour TensorRT 10.x, cuDNN 9.x, NCCL 2.x (déclarés mais pas quantifiés dans la source).

**Étapes :**

```bash
# 1. Prérequis : venv + CUDA 13.2
python3 -m venv /path/to/trtllm-env
source /path/to/trtllm-env/bin/activate

# 2. PyTorch pour CUDA 13.2
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Dépendances système (apt-get requis)
sudo apt-get install -y libopenmpi-dev libzmq3-dev

# 4. TensorRT-LLM — build from wheel (prébuild recommandé)
# À partir de la release NVIDIA :
pip install tensorrt-llm==1.3.0rc23 --index-url https://pypi.python.org/pypi

# OU build from source (si requis) :
# git clone https://github.com/NVIDIA/TensorRT-LLM.git
# cd TensorRT-LLM
# python3 scripts/build_wheel.py --cuda-architectures sm_120
# pip install build/tensorrt_llm-*.whl
```

**Avantages :** isolation, contrôle des versions, empreinte minimale.
**Inconvénients :** CMake + compilation CUDA locale requise si pas de wheel prébuild.

## Option 2 : Docker (multi-stage)

**Taille estimée :** 8–12 Go (release image seule, ~400–600 Mo compression).

**Stages utiles :**

| stage | contenu | taille (decomp.) |
|---|---|---|
| `base` | nvcr.io/nvidia/pytorch:26.05-py3 + OS pkgs | ~5 Go |
| `devel` | CUDA 13.2, CMake, TensorRT 10.x, cuDNN 9.x | +3 Go |
| `wheel` | build TensorRT-LLM wheel | ~1.5 Go (artifact) |
| `release` | runtime seul (stripped) | ~2.5 Go |

**Build :**

```bash
# À partir du dépôt NVIDIA
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
docker build -f docker/Dockerfile.multi --target release \
  --build-arg CUDA_ARCHITECTURES=sm_120 \
  -t trtllm:release-sm120 .
```

**Exécution :**

```bash
docker run --gpus all --rm -it trtllm:release-sm120 python3 -c \
  "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

**Avantages :** reproductibilité totale, CUDA préinstallé, multi-GPU ready (MPI, NCCL, UCX).
**Inconvénients :** taille, temps de build (~30–45 min CPU native), interopérabilité réseau complexe.

## Décision : venv recommandé

**Argument :** venv + wheel prébuild (si disponible) = 3–4 Go, 5–10 min setup, sans compilation locale. Docker gardé en option si parallélisation multi-GPU ou déploiement conteneurisé justifié plus tard.

**Prochaines étapes après validation utilisateur :**
1. Vérifier la disponibilité d'une wheel sm_120 prébuild v1.3.0rc23 sur PyPI/NVIDIA
2. Si oui : setup venv, pip install PyTorch 2.13.0 cu132 + tensorrt-llm, test `acvram doctor`
3. Si non : build venv depuis source (CMake, 30 min supplémentaires, CUDA_HOME=/usr/local/cuda-13.2)

---
À sec le 21/09 09:29 — sans téléchargement ni build. Pointeur : Maîtresse approuve / refuse la pièce 3.
