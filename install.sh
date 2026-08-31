#!/usr/bin/env bash
# Installateur acvram. Choisit la version de torch adaptee aux GPU presents.
set -euo pipefail

cd "$(dirname "$0")"
VENV="${ACVRAM_VENV:-.venv}"
PY="${PYTHON:-python3}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---- python -----------------------------------------------------------------
command -v "$PY" >/dev/null || die "python3 introuvable"
"$PY" - <<'EOF' || die "acvram exige Python 3.10 ou plus recent"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
EOF
say "python: $("$PY" --version)"

# ---- choix de l'index torch -----------------------------------------------------
# Le sm_120 (Blackwell) ne tourne pas sur une roue compilee pour CUDA < 12.8.
INDEX=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    say "capacites de calcul presentes : ${CAPS//$'\n'/, }"
    if grep -qE '^(12|10)\.' <<<"$CAPS"; then
        INDEX="https://download.pytorch.org/whl/cu130"
        NVCC_WHEEL=1
        say "Blackwell detecte -> installation de torch pour CUDA 13.0"
    else
        INDEX="https://download.pytorch.org/whl/cu124"
    fi
else
    warn "pas de nvidia-smi ; installation de torch pour processeur"
    INDEX="https://download.pytorch.org/whl/cpu"
fi

# ---- environnement virtuel -------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    say "creation de $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

say "installation de torch depuis $INDEX"
pip install --quiet --index-url "$INDEX" torch

# Le nvcc de la distribution est souvent trop ancien pour emettre du sm_120
# (Mint 22.3 livre CUDA 12.0). On prend celui des roues pip, que
# acvram/kernels/__init__.py sait trouver tout seul.
if [ "${NVCC_WHEEL:-0}" = 1 ]; then
    say "installation de nvcc (roues cuda-toolkit)"
    pip install --quiet --only-binary=:all: 'cuda-toolkit[nvcc]' \
        || warn "nvcc non installe ; les noyaux fusionnes retomberont sur la reference"
fi

say "installation d'acvram"
pip install --quiet -e '.[dev]'

# ---- verification -----------------------------------------------------------------
say ""
acvram doctor || warn "doctor a signale des problemes ; voir ci-dessus"
say ""
say "activez avec :  source $VENV/bin/activate"
say "puis essayez :   acvram detect"
