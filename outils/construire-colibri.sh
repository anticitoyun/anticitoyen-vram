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
#   Correctif TOPK_DUMP (bd 076, poste5, contre fd93c41) : outils/colibri/topk-dump.patch, appliqué
#   AVANT make — refus nommé si git apply --check échoue (clone sur un autre commit que fd93c41,
#   ou patch déjà intégré en amont) ; déjà appliqué (reprise) → pas réappliqué, pas d'erreur.
set -euo pipefail

FAMILLE="${1:-qwen36}"
COLIBRI_DIR="${COLIBRI_DIR:-/mnt/AI_GENERATOR/colibri}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PATCH="$(dirname "$(readlink -f "$0")")/colibri/topk-dump.patch"

c_d=$'\e[2m'; c_v=$'\e[32m'; c_r=$'\e[31m'; c_0=$'\e[0m'
[ -t 1 ] || { c_d=''; c_v=''; c_r=''; c_0=''; }
err() { printf '%s%s%s\n' "$c_r" "$*" "$c_0" >&2; }

[ -d "$COLIBRI_DIR/c" ] || { err "dossier colibri introuvable : $COLIBRI_DIR (COLIBRI_DIR=... pour un autre chemin)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { err "nvcc introuvable sous CUDA_HOME=$CUDA_HOME"; exit 1; }

if [ -f "$PATCH" ]; then
  if git -C "$COLIBRI_DIR" apply --check "$PATCH" 2>/dev/null; then
    printf '%sapplication du correctif TOPK_DUMP%s\n' "$c_d" "$c_0"
    git -C "$COLIBRI_DIR" apply "$PATCH"
  elif git -C "$COLIBRI_DIR" apply --check --reverse "$PATCH" 2>/dev/null; then
    printf '%scorrectif TOPK_DUMP déjà appliqué%s\n' "$c_d" "$c_0"
  else
    err "correctif TOPK_DUMP inapplicable sur $COLIBRI_DIR (autre commit que fd93c41 ? voir $PATCH)"; exit 1
  fi
else
  err "avertissement : $PATCH introuvable — construction sans TOPK_DUMP"
fi

printf '%sconstruction colibri : %s CUDA_ARCH=sm_120 (CUDA_HOME=%s)%s\n' "$c_d" "$FAMILLE" "$CUDA_HOME" "$c_0"
make -C "$COLIBRI_DIR/c" "$FAMILLE" CUDA=1 CUDA_ARCH=sm_120 CUDA_HOME="$CUDA_HOME"

BIN="$COLIBRI_DIR/c/$FAMILLE"
[ -x "$BIN" ] || { err "construction terminée mais binaire absent : $BIN"; exit 1; }
printf '%sconstruit : %s%s\n' "$c_v" "$BIN" "$c_0"
