#!/bin/bash
# Bras cassant de tests/test_exil_equivalence_gpu.py (poste7-exil-ppl-89-priorite-17-09) :
# retire le `wait` de layers.py:399 (StreamedWeight.attendre) et vérifie que le test
# devient rouge. Nécessite une carte (outils/carte.sh) — jamais lancé sans elle. Restaure
# par git checkout dans tous les cas (set -u, sortie normale ou signal).
set -u
R="acvram/engine/layers.py"
P=../../anticitoyen-vram/.venv/bin/python
restore() { git checkout -q -- "$R"; }
trap restore EXIT

echo "=== témoin (avec le wait, doit passer) ==="
outils/carte.sh env CUDA_VISIBLE_DEVICES=0 $P -m pytest tests/test_exil_equivalence_gpu.py -q

echo "=== retrait du wait (layers.py:399) ==="
sed -i '399s/.*/            pass  # casser-exil-wait-17-09 : wait retire/' "$R"
grep -n "casser-exil-wait" "$R"

echo "=== bras casse (sans le wait, doit devenir rouge) ==="
outils/carte.sh env CUDA_VISIBLE_DEVICES=0 $P -m pytest tests/test_exil_equivalence_gpu.py -q
