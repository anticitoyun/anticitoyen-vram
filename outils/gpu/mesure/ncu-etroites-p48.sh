#!/bin/bash
# Pièce 48 — où part le débit à M = 12 : trois bras ncu sur les noyaux EN L'ÉTAT
# (aucun code modifié). Prédictions et seuils : revue/poste1-piece48-ncu-etroites-22-09.md
#
#   outils/carte.sh outils/gpu/mesure/ncu-etroites-p48.sh <commit-attendu> [sortie]
#
# Bras A `_dense_etroit_kernel` (nvfp4) et C `nvfp4_gemv_marlin_kernel` sur
# l'alias alpha2 ; bras B `etroit`/`gemm_etroit` int8 sur l'alias officiel.
# ncu est lent (replay par noyau) : `--launch-count` borne chaque bras, les
# métriques sont ciblées — `--set full` coûterait la fenêtre entière.
set -uo pipefail
cd "$(dirname "$0")/../../.."
ATTENDU=${1:?usage: ncu-etroites-p48.sh <commit-attendu> [sortie]}
D=${2:-scratchpad/poste1-p48-ncu}
[ "$(git rev-parse --short HEAD)" = "$ATTENDU" ] || {
  echo "REFUS : HEAD $(git rev-parse --short HEAD) != $ATTENDU" >&2; exit 65; }
mkdir -p "$D"
export PYTHONPATH=$PWD
PY=$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python
ALPHA2=/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha2-22-09
OFFICIEL=/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4
M="smsp__issue_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
sm__inst_executed.sum,\
gpu__time_duration.sum"

echo "== compute-apps début $(date +%H:%M:%S)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
echo "== charge hôte"; ps -eo pid,pcpu,comm --sort=-pcpu | head -4

bras () {                       # $1 nom, $2 alias, $3 regex de noyau, $4 lancements
  echo "== bras $1 ($3) $(date +%H:%M:%S)"
  ACVRAM_MODELE_MESURE="$2" ncu --graph-profiling node -k regex:"$3" \
      --launch-count "$4" --metrics "$M" --csv \
      $PY outils/gpu/mesure/frontiere-pas.py "$D/$1.json" 12 8 \
      > "$D/$1.csv" 2> "$D/$1.log"
  local rc=$?
  echo "   rc=$rc  lignes=$(wc -l < "$D/$1.csv" 2>/dev/null || echo 0)"
  # ncu refuse parfois le profilage sous graphe : le dire, ne pas le masquer
  grep -qi "graph" "$D/$1.log" && grep -i "graph" "$D/$1.log" | head -2
  return 0
}

bras A_nvfp4_etroit   "$ALPHA2"  "_dense_etroit_kernel"     6
bras B_int8_etroit    "$OFFICIEL" "etroit|gemm_etroit"      6
bras C_marlin_avec    "$ALPHA2"  "nvfp4_gemv_marlin_kernel" 4
bras C_marlin_sans    "$OFFICIEL" "nvfp4_gemv_marlin_kernel" 4

echo "== compute-apps fin $(date +%H:%M:%S)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
echo "== résumé"
$PY outils/gpu/mesure/ncu-resume-p48.py "$D"
