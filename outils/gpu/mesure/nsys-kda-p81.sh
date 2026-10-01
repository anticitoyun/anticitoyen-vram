#!/usr/bin/env bash
# p81 : part du KDA dans le pas de Kimi-Linear-35B (préfill 8 k, décodage b=1 et b=12), une trace, un chargement.
# nsys DANS la prise (leçon du 22/09), --cuda-graph-trace=node obligatoire (un rejeu de graphe = ses nœuds).
#   ACVRAM_POSTE=poste1 ACVRAM_NOM=poste1-p81-nsys ACVRAM_TYPE=mesure ACVRAM_DUREE_MAX=600 ATTENDU=<HEAD> \
#       outils/carte.sh bash outils/gpu/mesure/nsys-kda-p81.sh
# Durée prédite : chargement ~1-2 min + 2 préfills de 8 k + 2 × 80 pas + post-traitement ≈ 4-6 min.
set -euo pipefail
ICI=$(cd "$(dirname "$0")" && pwd)
ARBRE=$(cd "$ICI/../../.." && pwd)
[ "$(git -C "$ARBRE" rev-parse HEAD)" = "${ATTENDU:?ATTENDU = HEAD figé}" ] || { echo "ÉCHEC : HEAD ≠ ATTENDU"; exit 65; }
PY=${PY_ACVRAM:-$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python}
export ACVRAM_MODELE_MESURE=${ACVRAM_MODELE_MESURE:-/mnt/AI_GENERATOR/models_acvram/Kimi-Linear-35B-kda-nvfp4}
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ARBRE" ACVRAM_ARBRE="${ACVRAM_ARBRE:-$ARBRE}"
O=${SORTIE:-$ARBRE/scratchpad/poste1-p81-nsys-$(date +%d-%m)}
mkdir -p "$O"
echo "[p81] $ACVRAM_MODELE_MESURE · HEAD $ATTENDU · sortie $O"
{ date +%FT%T; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; ps -eo pid,pcpu,comm --sort=-pcpu | head -4; } > "$O/avant.txt"
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o "$O/kda" --force-overwrite true \
    "$PY" "$ICI/kda-part-p81.py" piloter > "$O/pilote.log" 2>&1
grep -E 'régime|pilote terminé' "$O/pilote.log" | cut -c1-200
nsys stats --report nvtx_kern_sum --format csv -o "$O/kda" "$O/kda.nsys-rep" > "$O/stats.log" 2>&1
{ date +%FT%T; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$O/apres.txt"
"$PY" "$ICI/kda-part-p81.py" analyser "$O/kda_nvtx_kern_sum.csv" --json "$O/part.json" | tee "$O/part.txt"
echo "[p81] terminé — $O/part.txt"
