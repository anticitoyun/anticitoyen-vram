#!/usr/bin/env bash
# Octets DRAM reels et duree par noyau d'un pas de decodage (b sequences), sous ncu.
#   outils/carte.sh outils/ncu_pas_decodage.sh 12 [gemv|mma]
# Filtre : les noyaux du pas (MoE, INT8, attention, routage, glue, norme) ; on saute
# les premiers lancements (chargement + 2 passes de chauffe + prefill) et on
# profile un echantillon. Sortie CSV dans ${NCU_SORTIE}/ncu-pas-<bras>.csv
# puis un tableau agrege par noyau : appels, ms, Go DRAM lus, Go/s effectifs.
set -euo pipefail
# Dossier de sortie (CSV, journaux, HOME de travail pour root) : NCU_SORTIE, par defaut
# /tmp/ncu-acvram -- jamais un chemin de session en dur (cliquet identifiants).
NCU_SORTIE=${NCU_SORTIE:-/tmp/ncu-acvram}; mkdir -p "$NCU_SORTIE"
B=${1:-12}; BRAS=${2:-gemv}
ICI=$(dirname "$(readlink -f "$0")"); REPO=$(dirname "$ICI")
PY=${ACVRAM_PY:-$REPO/../../anticitoyen-vram/.venv/bin/python3}
OUT=${NCU_SORTIE}/ncu-pas-${BRAS}.csv
export ACVRAM_TYPE=mesure BANC_JETONS=8
/usr/local/cuda/bin/ncu --csv --target-processes all \
  --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name "regex:nvfp4_gemv_grouped|int8_gemv|paged_attn|moe_route|moe_reduce|rmsnorm|rms_norm|rope|nvfp4_gemm_grouped|silu|topk|elementwise|gather|index" \
  --launch-skip 4000 --launch-count 1200 \
  "$PY" "$ICI/banc_decodage_moe.py" "$BRAS" "$B" > "$OUT" 2>${NCU_SORTIE}/ncu-pas-${BRAS}.err || true
"$PY" - "$OUT" <<'PYEOF'
import csv, sys, collections
rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0].isdigit()]
# colonnes : ID, Process ID, Process Name, Host Name, Kernel Name, Context, Stream, Block Size, Grid Size, Device, CC, Section, Metric Name, Metric Unit, Metric Value
agg = collections.defaultdict(lambda: {"n": 0, "t": 0.0, "rd": 0.0, "wr": 0.0, "sm": 0.0})
seen = set()
for r in rows:
    kid, nom, met, unit, val = r[0], r[4], r[12], r[13], r[14].replace(",", "")
    try: v = float(val)
    except ValueError: continue
    k = nom.split("(")[0][:70]
    a = agg[k]
    if met == "gpu__time_duration.sum":
        a["t"] += v / (1e6 if unit == "nsecond" else 1e3 if unit == "usecond" else 1.0); a["n"] += 1
    elif met == "dram__bytes_op_read.sum": a["rd"] += v * {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(unit, 1)
    elif met == "dram__bytes_op_write.sum": a["wr"] += v * {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(unit, 1)
    elif met.startswith("sm__throughput"): a["sm"] += v
tot_t = sum(a["t"] for a in agg.values())
print(f"{'noyau':70s} {'appels':>6s} {'ms':>8s} {'%':>5s} {'Go lus':>7s} {'Go ecr':>7s} {'Go/s eff':>9s} {'SM%':>5s}")
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
    if a["n"] == 0: continue
    bw = (a["rd"] + a["wr"]) / (a["t"] / 1e3) / 1e9 if a["t"] > 0 else 0
    print(f"{k:70s} {a['n']:6d} {a['t']:8.3f} {100*a['t']/tot_t:5.1f} {a['rd']/1e9:7.3f} {a['wr']/1e9:7.3f} {bw:9.0f} {a['sm']/a['n']:5.1f}")
print(f"TOTAL {tot_t:.3f} ms sur {sum(a['n'] for a in agg.values())} lancements echantillonnes")
PYEOF
