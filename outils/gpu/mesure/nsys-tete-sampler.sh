#!/usr/bin/env bash
# Pièce 39 : part de la TÊTE et du SAMPLER-GRAPHE dans le pas servi b=12.
#
# La table de familles du 22/09 donne « tête 0,132 ms » et « échantillon
# 3,7 µs » (frontiere-pas) sur des prises différentes ; cette chaîne les
# mesure dans LA MÊME trace, en service réel, avec les défauts 0.6.35
# (sampler=graphe, rapatriement=epingle), et rend les deux parts en µs par
# pas et en % du pas.
#
# Leçon du 22/09 (verdict-nsys-familles) : nsys DANS la prise carte.sh, pas
# l'inverse — carte.sh enveloppe CE script, qui lance nsys lui-même ; et
# `--cuda-graph-trace=node` est obligatoire (sans lui, un rejeu de graphe
# compte pour un seul « noyau » et la granularité est fausse).
#
#   ACVRAM_NOM=nsys-tete ACVRAM_TYPE=mesure outils/carte.sh \
#       bash outils/gpu/mesure/nsys-tete-sampler.sh
#
# Durée : chargement + chauffe + 10 s de fenêtre + post-traitement ≈ 5 min.
set -u
ICI=$(cd "$(dirname "$0")" && pwd)
ARBRE=${ACVRAM_ARBRE:-$(cd "$ICI/../../.." && pwd)}
PY=${PY_ACVRAM:-$ARBRE/.venv/bin/python}
MODELE=${ACVRAM_MODELE_MESURE:-$(${PY} -c "import sys; sys.path.insert(0,'$ARBRE/outils'); from racine_modeles import racine_modeles as r; print(r()+'/Qwen3-Coder-30B-A3B-nvfp4')")}
O=${SORTIE:-$ICI/../../../scratchpad/nsys-tete-sampler-$(date +%d-%m)}
B=${B:-12}
PAS=${PAS:-60}
mkdir -p "$O"

echo "[nsys-tete] modèle $MODELE · b=$B · $PAS pas · sortie $O"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$O/avant.txt"

nsys profile -t cuda --cuda-graph-trace=node -o "$O/graphe" --force-overwrite true \
    "$PY" "$ICI/frontiere-pas.py" "$O/frontiere.json" "$B" "$PAS" > "$O/nsys.log" 2>&1
rc=$?
echo "[nsys-tete] nsys rc $rc"
nsys stats --report cuda_gpu_trace --format csv -o "$O/graphe" "$O/graphe.nsys-rep" >> "$O/nsys.log" 2>&1

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$O/apres.txt"
"$PY" "$ICI/familles-noyaux.py" "$O/graphe_cuda_gpu_trace.csv" --json "$O/familles.json" \
      --detail tete | tee "$O/familles.txt"
"$PY" "$ICI/part-tete-sampler.py" "$O/graphe_cuda_gpu_trace.csv" --json "$O/part.json" | tee "$O/part.txt"
echo "[nsys-tete] terminé — $O/part.txt"
