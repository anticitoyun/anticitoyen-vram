#!/bin/bash
# 296 (poste6) : lance ComfyUI 0.37.4 (env conda comfy_blackwell312) sur la carte reservee, rend les graphes video
# de parc/share/openwebui/videos/ par outils/videos-owui-rendu.py, puis arrete ComfyUI. A lancer SOUS carte.sh :
#   ACVRAM_NOM=poste6-296-video ACVRAM_DUREE_MAX=1800 outils/carte.sh outils/videos-owui-rendu.sh <tsv> [graphes...]
set -euo pipefail
TSV=${1:?tsv de sortie}; shift
COMFY=/mnt/AI_GENERATOR/Comfyuirtx5090/ComfyUI312/ComfyUI
# L'env conda de ComfyUI312 vit sous un autre compte : passer COMFY_ENV (racine de l'env) depuis le lanceur.
ENV=${COMFY_ENV:-$HOME/miniconda3/envs/comfy_blackwell312}
PY=$ENV/bin/python
_SP=$ENV/lib/python3.12/site-packages
[ -x "$PY" ] || { echo "python de ComfyUI introuvable : $PY (COMFY_ENV)" >&2; exit 65; }
export NUMEXPR_MAX_THREADS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/mnt/AI_GENERATOR/Comfyuirtx5090/.inductor_cache
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$_SP/nvidia/cublas/lib:$_SP/nvidia/cudnn/lib:$_SP/nvidia/cuda_nvrtc/lib:$_SP/nvidia/cufft/lib:$_SP/nvidia/curand/lib:$_SP/nvidia/cuda_runtime/lib"
ICI=$(cd "$(dirname "$0")/.." && pwd)
JOURNAL=${TSV%.tsv}.comfy.log
ss -ltn | grep -q ':8188 ' && { echo "port 8188 deja pris" >&2; exit 65; }
( cd "$COMFY" && exec "$PY" main.py --fast fp16_accumulation --use-poste7-attention --listen 127.0.0.1 --port 8188 --disable-auto-launch \
    --output-directory /mnt/2TO_SSD_2025_IA --temp-directory /mnt/2TO_SSD_2025_IA/0TEMP ) > "$JOURNAL" 2>&1 &
CPID=$!
trap 'kill "$CPID" 2>/dev/null; wait "$CPID" 2>/dev/null || true; echo "ComfyUI arrete ($CPID)"' EXIT
for i in $(seq 1 180); do ss -ltn | grep -q ':8188 ' && break; kill -0 "$CPID" 2>/dev/null || { echo "ComfyUI mort au demarrage, voir $JOURNAL" >&2; exit 1; }; sleep 2; done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/compute-apps debut: /'
python3 "$ICI/outils/videos-owui-rendu.py" --tsv "$TSV" ${1:+--seulement "$@"}
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/compute-apps fin: /'
