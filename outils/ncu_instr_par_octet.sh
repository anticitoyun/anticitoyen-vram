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
VLLM_MODELE=${VLLM_MODELE:-/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4}
export ACVRAM_TYPE=mesure BANC_PAS_NCU=${BANC_PAS_NCU:-1}
# Chaque noyau est rejoue par ncu avec sauvegarde/restauration de TOUTE la memoire
# allouee (15 Gio de poids) a chaque passe : peu de metriques (peu de passes), un
# seul pas (~600 noyaux). sm__throughput (composite, beaucoup de passes) est en
# option : NCU_METRIQUES=... pour l'ajouter.
case "$MOTEUR" in
  acvram) PY=${PY:-~/Bureau/Claude/anticitoyen-vram/.venv/bin/python3}
          export BANC_GRAPHES=${BANC_GRAPHES:-1} BANC_JETONS=8
          CMD=("$PY" "$ICI/banc_decodage_moe.py" ncu "$B") ;;
  vllm)   PY=/opt/ia/vLLM/.venv/bin/python; export BANC_SLOTS=$B
          CMD=("$PY" "$ICI/ncu_vllm_decode12.py" "$VLLM_MODELE") ;;
  *) echo "moteur inconnu : $MOTEUR"; exit 2 ;;
esac
OUT=/tmp/claude-1000/ncu-ipo-${MOTEUR}.csv
# Les compteurs sont reserves a root (RmProfilingAdminOnly=1) : sudoers NOPASSWD sur
# ncu (docs/MATERIEL.md). sudo remet l'environnement a zero -> on le repasse par env,
# avec un HOME de travail pour que root n'ecrive rien dans les caches de l'utilisateur.
HOME_NCU=${HOME_NCU:-/tmp/claude-1000/home-ncu}; mkdir -p "$HOME_NCU"
# Copie du cache de noyaux compile de CET arbre (cle = sha256 du chemin du paquet,
# acvram/kernels/__init__.py) pour que root ne recompile pas ni n'ecrive chez l'utilisateur.
REPO=$(dirname "$ICI"); CLE=$(python3 -c "import hashlib,os;print(hashlib.sha256(os.path.realpath('$REPO/acvram/kernels').encode()).hexdigest()[:12])")
KC="$HOME_NCU/kernels-$CLE"; [ -d "$KC" ] || cp -r "$HOME/.cache/acvram/kernels-$CLE" "$KC"
ENV=(env HOME="$HOME_NCU" XDG_CACHE_HOME="$HOME_NCU/.cache" TRITON_CACHE_DIR="$HOME_NCU/.triton" PATH="$PATH"
     CUDA_VISIBLE_DEVICES=0 ACVRAM_TYPE=mesure BANC_PAS_NCU="$BANC_PAS_NCU" BANC_GRAPHES="${BANC_GRAPHES:-1}" BANC_JETONS=8 BANC_SLOTS="$B"
     VLLM_ENABLE_V1_MULTIPROCESSING=0 ACVRAM_KERNEL_CACHE="$KC" ACVRAM_INT8_TRANCHE="${ACVRAM_INT8_TRANCHE:-16}")
# NCU_CACHE=none : pas de purge des caches entre les rejeux (octets DRAM tels que
# le pas les voit, L2 chaud entre deux noyaux voisins) ; defaut ncu = all (purge :
# chaque noyau est mesure a froid, ce qui compte en DRAM une relecture que le L2 sert
# en vrai -- lecon du 14/09 sur int8_gemv<4,4>).
sudo -n /usr/local/cuda/bin/ncu --csv --target-processes all --clock-control none --cache-control "${NCU_CACHE:-all}" \
  --nvtx --nvtx-include "mesure/" \
  --metrics "${NCU_METRIQUES:-gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__cycles_elapsed.avg.per_second}" \
  "${ENV[@]}" "${CMD[@]}" > "$OUT" 2>/tmp/claude-1000/ncu-ipo-${MOTEUR}.err || true
grep -c "^\"[0-9]" "$OUT" || { echo "ECHEC : rien profile, voir /tmp/claude-1000/ncu-ipo-${MOTEUR}.err"; tail -5 /tmp/claude-1000/ncu-ipo-${MOTEUR}.err; exit 1; }
python3 "$ICI/ncu_instr_par_octet_agrege.py" "$OUT" "$BANC_PAS_NCU" | tee /tmp/claude-1000/ncu-ipo-${MOTEUR}.txt
