#!/usr/bin/env bash
# Passe ncu BORNÉE (lecture seule, REGLES § 6) sur la GEMV groupée des experts
# (poste7-gemv-experts-rpw-18-09 : scellé RPW faux ⇒ lire avant d'écrire) :
# pourquoi 63 % de bande ? stalls (long_scoreboard = latence mémoire, barrier
# = __syncthreads de l'étage, lg/mio_throttle = files de chargement),
# occupation atteinte, limite d'occupation par la shared, L2 hit, secteurs par
# requête (coalescence), octets DRAM et durée — par noyau, sur UN pas.
#   ACVRAM_GROUPED_RPW=4 outils/carte.sh outils/ncu_gemv_experts_rpw_18-09.sh 12                # défaut
#   ACVRAM_GROUPED_XREG=1 outils/carte.sh outils/ncu_gemv_experts_rpw_18-09.sh 12               # dernier geste
# (un processus par valeur : le .cu fige RPW et XREG au premier lancement)
# Porte du dernier geste (poste7-gemv-experts-dernier-geste-18-09) : gateup + down ≤ 6,2 ms/pas
# (somme des « ×48 ») sous ncu, contre 7,35 à rpw=4 ; identique au bit (tests/test_gemv_experts_v2.py).
# Sortie : ${NCU_SORTIE}/ncu-gemv-rpw<N>-xreg<X>.csv + tableau agrégé (médiane par noyau).
set -euo pipefail
NCU_SORTIE=${NCU_SORTIE:-/tmp/ncu-acvram}; mkdir -p "$NCU_SORTIE"
B=${1:-12}; RPW=${ACVRAM_GROUPED_RPW:-4}
ICI=$(dirname "$(readlink -f "$0")"); REPO=$(dirname "$ICI")
PY=${ACVRAM_PY:-$REPO/../../anticitoyen-vram/.venv/bin/python3}
OUT=${NCU_SORTIE}/ncu-gemv-rpw${RPW}.csv
XREG=${ACVRAM_GROUPED_XREG:-down}
OUT=${NCU_SORTIE}/ncu-gemv-rpw${RPW}-xreg${XREG}.csv
# `sudo -n ncu` remet l'environnement à zéro (poste3, verdict-ncu-gemv-experts-rpw-
# 18-09 : aucune ACVRAM_* n'atteignait la cible, chemin mma, fg=0) et sudoers
# n'autorise NOPASSWD que sur le binaire ncu lui-même (`sudo -n env … ncu`
# rend rc=1) : le régime est posé dans un WRAPPER cible, exécuté par ncu.
NCU=${NCU:-$(ls /usr/local/cuda*/bin/ncu 2>/dev/null | sort -V | tail -1)}
WRAP=${NCU_SORTIE}/cible-rpw${RPW}-xreg${XREG}.sh
cat > "$WRAP" <<WEOF
#!/usr/bin/env bash
export ACVRAM_TYPE=mesure BANC_JETONS=8 ACVRAM_GROUPED_RPW=$RPW ACVRAM_GROUPED_XREG=$XREG
export LC_ALL=C HOME=$HOME PATH=$PATH PYTHONPATH=${PYTHONPATH:-}
exec "$PY" "$ICI/banc_decodage_moe.py" gemv "$B"
WEOF
chmod +x "$WRAP"
# 48 couches × 2 noyaux (gateup, down) = 96 lancements par pas ; on saute deux pas
# après le prefill (chauffe) et on profile UN pas. LC_ALL=C dans le wrapper : sortie
# ncu sans séparateurs de milliers ni virgules décimales (le parseur cassait en fr).
# dram__bytes_read.sum : n/a sur 5090 / ncu 2026.3, retiré — la bande par dram__throughput.
sudo -n "$NCU" --csv --target-processes all \
  --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,launch__waves_per_multiprocessor,launch__registers_per_thread,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  --kernel-name "regex:nvfp4_gemv_grouped_gateup_kernel|nvfp4_gemv_grouped_warp_kernel|nvfp4_gemv_grouped_gateup_xreg_kernel|nvfp4_gemv_grouped_xreg_kernel" \
  --launch-skip 192 --launch-count 96 \
  "$WRAP" > "$OUT" 2>${NCU_SORTIE}/ncu-gemv-rpw${RPW}-xreg${XREG}.err || true
"$PY" - "$OUT" "$RPW" "$XREG" <<'PYEOF'
import csv, sys, collections, statistics, re
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0].isdigit()]
if not rows:
    print("aucune ligne ncu : voir le .err (sudo -n ncu ? filtre ? launch-skip trop grand ?)"); sys.exit(1)

def nombre(v):
    # tolère la locale fr malgré LC_ALL=C : « 91 040 » (espace ou insécable de milliers), « 45,56 » (virgule décimale)
    v = re.sub(r"[\s\u00a0\u202f]", "", v)
    if "," in v and "." not in v:
        v = v.replace(",", ".")
    return float(v.replace(",", ""))
par = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    nom, met, val = r[4], r[12], r[14]
    try:
        par[nom.split("<")[0]][met].append(nombre(val))
    except ValueError:
        pass                                   # n/a
print(f"ACVRAM_GROUPED_RPW={sys.argv[2]} ACVRAM_GROUPED_XREG={sys.argv[3]} — médianes par noyau sur {len({r[0] for r in rows})} lancements")
total_pas = 0.0
for nom, mets in par.items():
    if "gpu__time_duration.sum" in mets:
        total_pas += statistics.median(mets["gpu__time_duration.sum"]) * 48 / 1e6
    print(f"\n{nom}")
    for met, vals in sorted(mets.items()):
        m = statistics.median(vals)
        if met == "gpu__time_duration.sum":
            print(f"  {met:78s} {m / 1e3:10.2f} µs  (× 48 = {m * 48 / 1e6:.2f} ms/pas par noyau)")
        elif met == "dram__bytes_read.sum":
            print(f"  {met:78s} {m / 1e6:10.2f} Mo")
        else:
            print(f"  {met:78s} {m:10.3f}")
    s = mets.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"); q = mets.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
    if s and q:
        print(f"  secteurs par requête (coalescence ; 4 = parfait pour 16 o/voie)            {statistics.median(s) / max(1, statistics.median(q)):10.2f}")
print(f"\ngateup + down : {total_pas:.2f} ms/pas (porte du dernier geste : ≤ 6,2 ; rpw=4 partagé : 7,35)")
PYEOF
echo "CSV : $OUT"
