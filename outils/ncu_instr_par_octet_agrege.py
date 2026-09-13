#!/usr/bin/env python3
"""Agrège le CSV ncu de outils/ncu_instr_par_octet.sh : par noyau (nom + grille
pour int8_gemv, dont la plus grande grille est lm_head) puis par poste.
inst/octet = instructions warp exécutées (sm__inst_executed) par octet DRAM lu."""
import collections
import csv
import re
import sys

rows = [r for r in csv.reader(open(sys.argv[1])) if r and r[0].isdigit()]
pas = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MULT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "inst": 1, "Kinst": 1e3, "Minst": 1e6, "Ginst": 1e9,
        "hz": 1, "Khz": 1e3, "Mhz": 1e6, "Ghz": 1e9, "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1}
par_lancement = collections.defaultdict(dict)
for r in rows:
    kid, nom, grille, met, unit, val = r[0], r[4], r[8], r[12], r[13], r[14].replace(",", "")
    try:
        v = float(val)
    except ValueError:
        continue
    d = par_lancement[kid]
    d["nom"], d["grille"] = nom, grille
    d[met] = v * MULT.get(unit, 1)

def cle(d):
    n = d["nom"].split("(")[0].split("<")[0][-60:]
    if "int8_gemv" in n:
        return f"{n} grille={d['grille']}"
    return n

agg = collections.defaultdict(lambda: collections.defaultdict(float))
for d in par_lancement.values():
    a = agg[cle(d)]
    a["n"] += 1
    for m in ("gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum", "sm__inst_executed.sum",
              "sm__inst_executed_pipe_tensor.sum", "sm__inst_executed_pipe_fma.sum", "sm__inst_executed_pipe_lsu.sum",
              "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__cycles_elapsed.avg.per_second"):
        a[m] += d.get(m, 0.0)

# La plus grande grille int8_gemv = lm_head (N = vocabulaire).
int8 = [k for k in agg if "int8_gemv" in k]
if int8:
    def gx(k):
        m = re.search(r"grille=\((\d+)", k); return int(m.group(1)) if m else 0
    lm = max(int8, key=gx)
    agg["int8_gemv lm_head " + lm.split(" grille=")[1]] = agg.pop(lm)

POSTES = [
    ("MoE gate·up", r"gateup|GroupProblemShape.*gate|grouped.*gate"),
    ("MoE down", r"nvfp4_gemv_grouped_warp|nvfp4_gemv_grouped_down"),
    ("MoE GEMM groupée (vLLM)", r"GroupProblemShape|grouped_gemm|GroupedGemm|Grouped"),
    ("lm_head", r"lm_head|wmma"),
    ("projections attention", r"int8_gemv|cutlass.*(fp4|e2m1)|sm120|Sm120|nvfp4_gemm|fp4_gemm|scaled_mm"),
    ("attention paginée", r"paged_attn|unified_attention|paged_attention|flash|fmha"),
    ("routage + glue MoE", r"moe_route|moe_reduce|moe_act|topk|shuffleInputRows|reduce_kernel|cvt_fp16_to_fp4|moe_align|sigmoid|softmax|grouped_topk|silu|expand|unpermute|permute|sort|cumsum|bincount|scatter|gather|index|Sort|sum_kernel|quant"),
    ("norm + rope + élémentaires", r"rmsnorm|rms_norm|layer_norm|rope|rotary|elementwise|vectorized|fill|copy|cat|add|mul|Copy|Fill"),
]
def poste(k):
    for p, rx in POSTES:
        if re.search(rx, k):
            return p
    return "reste"

def ligne(k, a, w):
    t = a["gpu__time_duration.sum"] / pas * 1e3
    rd = a["dram__bytes_read.sum"] / pas
    inst = a["sm__inst_executed.sum"] / pas
    ipo = inst / rd if rd else float("inf")
    bw = (a["dram__bytes_read.sum"] + a["dram__bytes_write.sum"]) / a["gpu__time_duration.sum"] / 1e9 if a["gpu__time_duration.sum"] else 0
    n = a["n"]
    return (f"{k[:w]:{w}s} {n/pas:7.1f} {t:8.3f} {rd/1e9:7.3f} {bw:7.0f} {inst/1e9:8.3f} {ipo:9.2f} "
            f"{100*a['sm__inst_executed_pipe_tensor.sum']/max(inst*pas,1):5.1f} {100*a['sm__inst_executed_pipe_fma.sum']/max(inst*pas,1):5.1f} "
            f"{100*a['sm__inst_executed_pipe_lsu.sum']/max(inst*pas,1):5.1f} {a['sm__throughput.avg.pct_of_peak_sustained_elapsed']/n:5.1f} "
            f"{a['sm__cycles_elapsed.avg.per_second']/n/1e6:5.0f}")

W = 64
ent = f"{'':{W}s} {'appels':>7s} {'ms/pas':>8s} {'Go lus':>7s} {'Go/s':>7s} {'Ginst':>8s} {'inst/oct':>9s} {'tens%':>5s} {'fma%':>5s} {'lsu%':>5s} {'SM%':>5s} {'MHz':>5s}"
print(f"Par pas (moyenne sur {pas} pas), {len(par_lancement)} lancements profilés")
print(ent)
tot_t = sum(a["gpu__time_duration.sum"] for a in agg.values())
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["gpu__time_duration.sum"])[:28]:
    print(ligne(k, a, W))
print()
print("Par poste")
print(ent)
pp = collections.defaultdict(lambda: collections.defaultdict(float))
for k, a in agg.items():
    p = pp[poste(k)]
    for m, v in a.items():
        p[m] += v
for k, a in sorted(pp.items(), key=lambda kv: -kv[1]["gpu__time_duration.sum"]):
    print(ligne(k, a, W) + f"  {100*a['gpu__time_duration.sum']/tot_t:5.1f} %")
tot = collections.defaultdict(float)
for a in agg.values():
    for m, v in a.items():
        tot[m] += v
print(ligne("TOTAL", tot, W))
