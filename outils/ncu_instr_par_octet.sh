#!/usr/bin/env bash
# Instructions par octet DRAM, par noyau, d'un pas de decodage b=12 : acvram ou vLLM.
#   outils/carte.sh outils/ncu_instr_par_octet.sh acvram [b]     (BANC_GRAPHES=1, rejeu)
#   outils/carte.sh outils/ncu_instr_par_octet.sh vllm   [b]     (moteur direct, /opt/ia/vLLM)
# Seuls les noyaux de la plage NVTX "mesure" (BANC_PAS_NCU pas de decodage, prefill
# exclu) sont profiles ; horloges non verrouillees (--clock-control none) pour que
# la duree soit celle du regime reel — les COMPTES (inst, octets) n'en dependent pas.
# Sortie : CSV brut + tableau agrege par noyau, puis par poste, dans
# ${NCU_SORTIE}/ncu-ipo-<moteur>.{csv,txt}
set -euo pipefail
# Dossier de sortie (CSV, journaux, HOME de travail pour root) : NCU_SORTIE, par defaut
# /tmp/ncu-acvram -- jamais un chemin de session en dur (cliquet identifiants).
NCU_SORTIE=${NCU_SORTIE:-/tmp/ncu-acvram}; mkdir -p "$NCU_SORTIE"
MOTEUR=${1:-acvram}; B=${2:-12}
ICI=$(dirname "$(readlink -f "$0")")
REPO=$(git -C "$ICI" rev-parse --show-toplevel 2>/dev/null || dirname "$ICI")
MODELE=${BANC_MODELE_CHEMIN:-$("$(dirname "$0")/racine_modeles.py")/Qwen3-Coder-30B-A3B-nvfp4}
VLLM_MODELE=${VLLM_MODELE:-/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4}
export ACVRAM_TYPE=mesure BANC_PAS_NCU=${BANC_PAS_NCU:-1}
# Chaque noyau est rejoue par ncu avec sauvegarde/restauration de TOUTE la memoire
# allouee (15 Gio de poids) a chaque passe : peu de metriques (peu de passes), un
# seul pas (~600 noyaux). sm__throughput (composite, beaucoup de passes) est en
# option : NCU_METRIQUES=... pour l'ajouter.
case "$MOTEUR" in
  acvram) PY=${ACVRAM_PY:-$REPO/../../anticitoyen-vram/.venv/bin/python3}
          export BANC_GRAPHES=${BANC_GRAPHES:-1} BANC_JETONS=8
          CMD=("$PY" "$ICI/banc_decodage_moe.py" ncu "$B") ;;
  vllm)   PY=/opt/ia/vLLM/.venv/bin/python; export BANC_SLOTS=$B
          CMD=("$PY" "$ICI/ncu_vllm_decode12.py" "$VLLM_MODELE") ;;
  *) echo "moteur inconnu : $MOTEUR"; exit 2 ;;
esac
OUT=${NCU_SORTIE}/ncu-ipo-${MOTEUR}.csv
# Les compteurs sont reserves a root (RmProfilingAdminOnly=1) : sudoers NOPASSWD sur
# ncu (docs/MATERIEL.md). sudo remet l'environnement a zero -> on le repasse par env,
# avec un HOME de travail pour que root n'ecrive rien dans les caches de l'utilisateur.
HOME_NCU=${HOME_NCU:-${NCU_SORTIE}/home-ncu}; mkdir -p "$HOME_NCU"
# Copie du cache de noyaux compile de CET arbre (cle = sha256 du chemin du paquet,
# acvram/kernels/__init__.py) pour que root ne recompile pas ni n'ecrive chez l'utilisateur.
REPO=$(dirname "$ICI"); CLE=$(python3 -c "import hashlib,os;print(hashlib.sha256(os.path.realpath('$REPO/acvram/kernels').encode()).hexdigest()[:12])")
KC="$HOME_NCU/kernels-$CLE"; [ -d "$KC" ] || cp -r "$HOME/.cache/acvram/kernels-$CLE" "$KC"
ENV=(env HOME="$HOME_NCU" XDG_CACHE_HOME="$HOME_NCU/.cache" TRITON_CACHE_DIR="$HOME_NCU/.triton" PATH="$PATH"
     CUDA_VISIBLE_DEVICES=0 ACVRAM_TYPE=mesure BANC_PAS_NCU="$BANC_PAS_NCU" BANC_GRAPHES="${BANC_GRAPHES:-1}" BANC_JETONS=8 BANC_SLOTS="$B"
     VLLM_ENABLE_V1_MULTIPROCESSING=0 ACVRAM_KERNEL_CACHE="$KC" ACVRAM_INT8_TRANCHE="${ACVRAM_INT8_TRANCHE:-16}"
     ACVRAM_CUDA_HOME="${ACVRAM_CUDA_HOME:-/usr/local/cuda-13.4}")
# ACVRAM_CUDA_HOME repasse a root : sans lui, torch regenere build.ninja avec le nvcc des
# roues pip (sans nv/target) -> ninja voit une commande differente -> RECOMPILATION sous ncu,
# qui echoue (19/09, M1 : « rien profile », extension en repli reference)
# Reglages du chemin MoE (bras mma) : repasses seulement s'ils sont poses.
for v in ACVRAM_NARROW_GEMM ACVRAM_NARROW_ROWS ACVRAM_NARROW_MIN_M ACVRAM_NVFP4_GEMV_MAX BANC_MODELE ACVRAM_MODELES ACVRAM_MOE_DECODE_FUSED ACVRAM_MOE_FUSED_TN ACVRAM_MOE_FUSED_ATOMIQUE ACVRAM_MOE_FUSED_ETAGES ACVRAM_MOE_MMA ACVRAM_MOE_GROUPED_MAX ACVRAM_MOE_MMA_BT ACVRAM_MOE_MMA_ETAGES ACVRAM_MOE_MMA_KS ACVRAM_MOE_GEMM_MAX ACVRAM_MOE_DECODE_MMA ACVRAM_MOE_DECODE_MMA_MIN_T ACVRAM_MOE_ROUTE_PACK; do
  [ -n "${!v:-}" ] && ENV+=("$v=${!v}")
done
# NCU_CACHE=none : pas de purge des caches entre les rejeux (octets DRAM tels que
# le pas les voit, L2 chaud entre deux noyaux voisins) ; defaut ncu = all (purge :
# chaque noyau est mesure a froid, ce qui compte en DRAM une relecture que le L2 sert
# en vrai -- lecon du 14/09 sur int8_gemv<4,4>).
# NCU_LANCEMENTS : plafond de noyaux profiles (defaut 2000 ~ un pas). ncu sauvegarde
# et restaure les 15 Gio du processus a chaque rejeu : ~0,75 s par noyau ; sans
# plafond, un pas du chemin MMA (3 617 lancements) prend 46 min (15/09). Une seule
# passe par appel de carte.sh : le processus est root (sudoers), il ne se tue pas.
sudo -n /usr/local/cuda/bin/ncu --csv --target-processes all --clock-control none --cache-control "${NCU_CACHE:-all}" \
  --launch-count "${NCU_LANCEMENTS:-2000}" \
  --nvtx --nvtx-include "mesure/" ${NCU_NOYAUX:+--kernel-name "regex:$NCU_NOYAUX"} \
  --metrics "${NCU_METRIQUES:-gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,sm__inst_executed.sum,sm__inst_executed_pipe_tensor.sum,sm__cycles_elapsed.avg.per_second}" \
  "${ENV[@]}" "${CMD[@]}" > "$OUT" 2>${NCU_SORTIE}/ncu-ipo-${MOTEUR}.err || true
grep -c "^\"[0-9]" "$OUT" || { echo "ECHEC : rien profile, voir ${NCU_SORTIE}/ncu-ipo-${MOTEUR}.err"; tail -5 ${NCU_SORTIE}/ncu-ipo-${MOTEUR}.err; exit 1; }
python3 "$ICI/ncu_instr_par_octet_agrege.py" "$OUT" "$BANC_PAS_NCU" | tee ${NCU_SORTIE}/ncu-ipo-${MOTEUR}.txt
