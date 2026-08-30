#!/usr/bin/env bash
# acvram installer. Picks the right torch build for the GPUs actually present.
set -euo pipefail

cd "$(dirname "$0")"
VENV="${ACVRAM_VENV:-.venv}"
PY="${PYTHON:-python3}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---- python -----------------------------------------------------------------
command -v "$PY" >/dev/null || die "python3 not found"
"$PY" - <<'EOF' || die "acvram needs Python 3.10 or newer"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
EOF
say "python: $("$PY" --version)"

# ---- pick a torch index -----------------------------------------------------
# sm_120 (Blackwell) cannot run on a wheel built against CUDA < 12.8.
INDEX=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    say "compute capabilities present: ${CAPS//$'\n'/, }"
    if grep -qE '^(12|10)\.' <<<"$CAPS"; then
        INDEX="https://download.pytorch.org/whl/cu128"
        say "Blackwell detected -> installing torch for CUDA 12.8"
    else
        INDEX="https://download.pytorch.org/whl/cu124"
    fi
else
    warn "no nvidia-smi; installing the CPU build of torch"
    INDEX="https://download.pytorch.org/whl/cpu"
fi

# ---- venv -------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    say "creating $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

say "installing torch from $INDEX"
pip install --quiet --index-url "$INDEX" torch

say "installing acvram"
pip install --quiet -e '.[dev]'

# ---- verify -----------------------------------------------------------------
say ""
acvram doctor || warn "doctor reported problems; see above"
say ""
say "activate with:  source $VENV/bin/activate"
say "then try:       acvram detect"
