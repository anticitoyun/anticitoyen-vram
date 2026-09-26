#!/bin/bash
# A/B PIPELINE=0 vs =1 en DEUX PROCESSUS SÉPARÉS (bead runner, 14/09 soir) :
# charger le modèle deux fois dans le MÊME processus (comportement d'origine
# de mesure-pipeline-ab.py) laisse une deuxième charge dans un état VRAM que
# `empty_cache()` ne récupère pas entièrement — constaté une fois avec un
# plan dégradé (exil, graphes CUDA désactivés). Confondu écarté depuis (deux
# processus donnent le même verdict que le run à un seul processus), mais ce
# script reste la mesure propre de référence.
#
#     outils/carte.sh outils/mesure-pipeline-ab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

"$PY" outils/mesure-pipeline-ab.py --seul 0
"$PY" outils/mesure-pipeline-ab.py --seul 1
