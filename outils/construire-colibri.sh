#!/usr/bin/env bash
# construire-colibri.sh — compile le moteur colibri (JustVugg/colibri, bd anticitoyen-vram-076)
# avec le palier CUDA de VRAM pour sm_120 (RTX 50-series). Compilation SEULEMENT : nvcc ne prend
# pas le GPU (il génère du SASS/PTX sur le CPU hôte), donc aucun verrou carte.sh ici — le
# lancement du binaire produit passe par parc/bin/colibri-serveur, lui sous carte.sh.
#
# Usage : construire-colibri.sh [famille]   (défaut : qwen36, seule famille servie aujourd'hui)
#   COLIBRI_DIR=/mnt/AI_GENERATOR/colibri (défaut) : racine du dépôt cloné.
#   CUDA_HOME=/usr/local/cuda (défaut) : /usr/bin/nvcc (paquet Debian, 12.0) passe avant le
#   toolkit 13.x dans le PATH d'un shell non interactif — même piège que acvram/kernels/__init__.py
#   (REGLES § pièges payés, chef.md) : on fixe CUDA_HOME, jamais le nvcc du PATH.
set -euo pipefail

FAMILLE="${1:-qwen36}"
COLIBRI_DIR="${COLIBRI_DIR:-/mnt/AI_GENERATOR/colibri}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

c_d=$'\e[2m'; c_v=$'\e[32m'; c_r=$'\e[31m'; c_0=$'\e[0m'
[ -t 1 ] || { c_d=''; c_v=''; c_r=''; c_0=''; }
err() { printf '%s%s%s\n' "$c_r" "$*" "$c_0" >&2; }

[ -d "$COLIBRI_DIR/c" ] || { err "dossier colibri introuvable : $COLIBRI_DIR (COLIBRI_DIR=... pour un autre chemin)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { err "nvcc introuvable sous CUDA_HOME=$CUDA_HOME"; exit 1; }

printf '%sconstruction colibri : %s CUDA_ARCH=sm_120 (CUDA_HOME=%s)%s\n' "$c_d" "$FAMILLE" "$CUDA_HOME" "$c_0"
make -C "$COLIBRI_DIR/c" "$FAMILLE" CUDA=1 CUDA_ARCH=sm_120 CUDA_HOME="$CUDA_HOME"

BIN="$COLIBRI_DIR/c/$FAMILLE"
[ -x "$BIN" ] || { err "construction terminée mais binaire absent : $BIN"; exit 1; }
printf '%sconstruit : %s%s\n' "$c_v" "$BIN" "$c_0"
