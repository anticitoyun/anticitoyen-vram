#!/bin/bash
# Deux processus separes (voir docstring de diag-logits-arrivee-jeton0.py).
#     outils/carte.sh outils/diag-logits-arrivee-jeton0.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

FE=$(mktemp); FG=$(mktemp)
"$PY" outils/diag-logits-arrivee-jeton0.py eager "$FE"
"$PY" outils/diag-logits-arrivee-jeton0.py graphes "$FG"
"$PY" - "$FE" "$FG" <<'EOF'
import json, sys
e, g = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ve, ie = e["vals"], e["idx"]
vg, ig = g["vals"], g["idx"]
print(f"\neager  : pas={e['pas']} top1={ie[0]} val={ve[0]:.6f}  top2={ie[1]} val={ve[1]:.6f}  ecart={ve[0]-ve[1]:.6f}")
print(f"graphes: pas={g['pas']} top1={ig[0]} val={vg[0]:.6f}  top2={ig[1]} val={vg[1]:.6f}  ecart={vg[0]-vg[1]:.6f}")
if ie[0] != ig[0]:
    print(f"candidats top1 DIFFERENTS : {ie[0]} (eager) vs {ig[0]} (graphes)")
EOF
