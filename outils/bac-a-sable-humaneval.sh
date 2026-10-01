#!/bin/bash
# e50.3 § 2 bis (précision chef, 01/10) — exécute un script Python généré par un modèle
# (HumanEval, e50.3) dans un bac à sable bwrap : réseau coupé, racine en lecture seule, $HOME
# masqué, seul /tmp (tmpfs neuf, détruit à la sortie) est inscriptible — un code généré
# n'est JAMAIS exécuté nu sur ce poste.
#
# Usage : bac-a-sable-humaneval.sh <script.py>
#   rc 0 si le script s'exécute SANS lever (le script lui-même décide de sa propre réussite,
#   p.ex. des `assert` de test ajoutés par l'appelant) ; rc != 0 sinon (échec du script, ou
#   une des bornes ci-dessous qui a coupé court — bwrap et les ulimits ne distinguent pas la
#   cause, lm-eval/qualite-e50.sh ne lisent que le rc).
#
# Bornes, par item :
#   mur 10 s (timeout), CPU 5 s (ulimit -t), 2 Gio d'espace d'adresses (ulimit -v),
#   64 processus/threads (ulimit -u), fichiers écrits <= 64 Mio (ulimit -f).
set -euo pipefail
SCRIPT=${1:?usage: bac-a-sable-humaneval.sh <script.py>}
[ -f "$SCRIPT" ] || { echo "REFUS : script introuvable ($SCRIPT)" >&2; exit 66; }
SCRIPT=$(readlink -f "$SCRIPT")
command -v bwrap >/dev/null || { echo "REFUS : bwrap absent (bubblewrap non installé)" >&2; exit 66; }

ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PY_BAC_A_SABLE : pour les tests seulement (tests/test_bac_a_sable_humaneval.py) — un
# interpréteur de test sans lm-eval, jamais posé par qualite-e50.sh en usage réel.
PY="${PY_BAC_A_SABLE:-$ICI/.venv-panel/bin/python}"
[ -x "$PY" ] || { echo "REFUS : $PY absent — installer .venv-panel d'abord (voir outils/panel-taches.sh)" >&2; exit 66; }

# $HOME est démonté (tmpfs) pour que le script généré n'y lise/écrive rien de réel — mais
# l'interpréteur `.venv-panel` vit SOUS $HOME (tous les worktrees y vivent) : sans le re-lier
# en lecture seule APRÈS le tmpfs, bwrap ne le verrait plus et ne pourrait pas s'exécuter
# lui-même. L'ordre des binds compte : celui-ci doit venir après `--tmpfs "$HOME"`.
exec timeout 10 bwrap \
  --unshare-all \
  --ro-bind / / \
  --tmpfs "$HOME" \
  --ro-bind "$ICI" "$ICI" \
  --remount-ro "$HOME" \
  --tmpfs /tmp \
  --bind "$SCRIPT" /tmp/essai.py \
  --die-with-parent \
  --new-session \
  --chdir /tmp \
  -- bash -c '
    ulimit -t 5
    ulimit -v 2097152
    ulimit -u 64
    ulimit -f 65536
    exec "'"$PY"'" /tmp/essai.py
  '
