#!/usr/bin/env bash
# Instructions par octet DRAM, par noyau, d'un pas de decodage b=12 : acvram ou vLLM.
#   outils/carte.sh outils/ncu_instr_par_octet.sh acvram [b]     (BANC_GRAPHES=1, rejeu)
#   outils/carte.sh outils/ncu_instr_par_octet.sh vllm   [b]     (moteur direct, /opt/ia/vLLM)
# Seuls les noyaux de la plage NVTX "mesure" (BANC_PAS_NCU pas de decodage, prefill
# exclu) sont profiles ; horloges non verrouillees (--clock-control none) pour que
# la duree soit celle du regime reel — les COMPTES (inst, octets) n'en dependent pas.
# Sortie : CSV brut + tableau agrege par noyau, puis par poste, dans
# /tmp/claude-1000/ncu-ipo-<moteur>.{csv,txt}
set -euo pipefail
MOTEUR=${1:-acvram}; B=${2:-12}
ICI=$(dirname "$(readlink -f "$0")")
MODELE=${BANC_MODELE_CHEMIN:-/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-nvfp4}
export ACVRAM_TYPE=mesure BANC_PAS_NCU=${BANC_PAS_NCU:-4}
case "$MOTEUR" in
  acvram) PY=${PY:-~/Bureau/Claude/anticitoyen-vram/.venv/bin/python3}
          export BANC_GRAPHES=${BANC_GRAPHES:-1} BANC_JETONS=8
          CMD=("$PY" "$ICI/banc_decodage_moe.py" ncu "$B") ;;
  vllm)   PY=/opt/ia/vLLM/.venv/bin/python; export BANC_SLOTS=$B
          CMD=("$PY" "$ICI/ncu_vllm_decode12.py" "${VLLM_MODELE:-$MODELE}") ;;
  *) echo "moteur inconnu : $MOTEUR"; exit 2 ;;
esac
OUT=/tmp/claude-1000/ncu-ipo-${MOTEUR}.csv
/usr/local/cuda/bin/ncu --csv --target-processes all --clock-control none \
  --nvtx --nvtx-include "mesure/" \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_lsu.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.per_second \
  "${CMD[@]}" > "$OUT" 2>/tmp/claude-1000/ncu-ipo-${MOTEUR}.err || true
grep -c "^\"[0-9]" "$OUT" || { echo "ECHEC : rien profile, voir /tmp/claude-1000/ncu-ipo-${MOTEUR}.err"; tail -5 /tmp/claude-1000/ncu-ipo-${MOTEUR}.err; exit 1; }
python3 "$ICI/ncu_instr_par_octet_agrege.py" "$OUT" "$BANC_PAS_NCU" | tee /tmp/claude-1000/ncu-ipo-${MOTEUR}.txt
