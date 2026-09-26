#!/bin/bash
# Lot poste 20/09/2026 (revue/poste7-poste-lot-20-09) : imprime ce qui a PRIS, clé=valeur, rien de mémoire.
# Usage : bash outils/poste/verifier.sh [pid du servant]   (sans pid : cherche le port 8090)
# Les valeurs sont lues sur le poste ; LnkCtl des GPU exige sudo lspci (sinon « (sudo) »).
set -u
pid="${1:-$(ss -tlnp 2>/dev/null | grep ':8090 ' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)}"
echo "date=$(date +%F_%H:%M:%S) uptime_min=$(( $(cut -d. -f1 /proc/uptime) / 60 ))"
echo "cmdline=$(cat /proc/cmdline)"
echo "max_cstate=$(cat /sys/module/intel_idle/parameters/max_cstate 2>/dev/null || echo absent)"
echo "cpuidle_etats_on=$(grep -L 1 /sys/devices/system/cpu/cpu0/cpuidle/state*/disable 2>/dev/null | wc -l) cpuidle_etats_off=$(grep -l 1 /sys/devices/system/cpu/cpu0/cpuidle/state*/disable 2>/dev/null | wc -l)"
echo "aspm_policy=$(cat /sys/module/pcie_aspm/parameters/policy 2>/dev/null || echo absent)"
for g in $(lspci -d 10de: 2>/dev/null | grep -Ei 'VGA|3D' | cut -d' ' -f1); do
  lnk=$(sudo -n lspci -vv -s "$g" 2>/dev/null | grep -E 'LnkSta:|LnkCtl:' | sed 's/^\s*//' | tr '\n' ' ')
  echo "gpu_$g=${lnk:-(sudo)}"
done
echo "thp_enabled=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "anonhuge_systeme_kB=$(grep AnonHugePages /proc/meminfo | awk '{print $2}')"
echo "env_session: ACVRAM_MODELES=${ACVRAM_MODELES:-} ACVRAM_MODELS_DIR=${ACVRAM_MODELS_DIR:-} THP_MEM_ALLOC_ENABLE=${THP_MEM_ALLOC_ENABLE:-} OMP_NUM_THREADS=${OMP_NUM_THREADS:-} ACVRAM_CPUS=${ACVRAM_CPUS:-}"
if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
  echo "servant_pid=$pid fils=$(ls /proc/$pid/task | wc -l) affinite=$(taskset -p "$pid" 2>/dev/null | awk '{print $NF}')"
  echo "servant_anonhuge_kB=$(grep AnonHugePages /proc/$pid/smaps_rollup 2>/dev/null | awk '{print $2}')"
  echo "servant_env: $(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(ACVRAM_MODELES|ACVRAM_MODELS_DIR|THP_MEM_ALLOC_ENABLE|OMP_NUM_THREADS|ACVRAM_CPUS)=' | tr '\n' ' ')"
else
  echo "servant_pid=aucun (port 8090 libre)"
fi
dest="${ACVRAM_MODELES:-/mnt/AI_GENERATOR/models_acvram}"
liste="$(dirname "$0")/alias-servis-20-09.txt"
if [ -f "$liste" ]; then
  manquants=0; total=0
  while read -r a; do
    case "$a" in ''|'#'*) continue ;; esac
    total=$((total+1)); [ -f "$dest/$a/acvram_manifest.json" ] || { manquants=$((manquants+1)); echo "alias_absent=$a"; }
  done < "$liste"
  echo "alias_total=$total alias_absents=$manquants dest=$dest"
fi
