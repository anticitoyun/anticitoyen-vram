#!/usr/bin/env python3
"""Agrège le CSV ncu de outils/ncu_instr_par_octet.sh : par noyau (nom + grille
pour int8_gemv, dont la plus grande grille est lm_head) puis par poste.
inst/octet = instructions warp exécutées (sm__inst_executed) par octet DRAM lu."""
import collections
import csv
import re
import sys

# L'en-tête n'est pas la première ligne (sortie du programme mêlée au CSV) et
# --nvtx insère des colonnes : on lit les colonnes par leur NOM.
lignes = [r for r in csv.reader(open(sys.argv[1]))]
i_ent = next(i for i, r in enumerate(lignes) if r and r[0] == "ID")
ent = lignes[i_ent]
col = {n: i for i, n in enumerate(ent)}
rows = [r for r in lignes[i_ent + 1:] if r and r[0].isdigit() and len(r) == len(ent)]
pas = int(sys.argv[2]) if len(sys.argv) > 2 else 1
illisibles = 0
MULT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "inst": 1, "Kinst": 1e3, "Minst": 1e6, "Ginst": 1e9,
        "hz": 1, "Khz": 1e3, "Mhz": 1e6, "Ghz": 1e9, "ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6,
        "ms": 1e-3, "msecond": 1e-3, "s": 1, "second": 1, "%": 1}
par_lancement = collections.defaultdict(dict)
for r in rows:
    kid, nom, grille = r[col["ID"]], r[col["Kernel Name"]], r[col["Grid Size"]]
    # Locale française : espaces (ou fines) pour les milliers, virgule décimale.
    met, unit, val = r[col["Metric Name"]], r[col["Metric Unit"]], r[col["Metric Value"]]
    val = val.replace(" ", "").replace("\u202f", "").replace("\xa0", "").replace(",", ".")
    try:
        v = float(val)
    except ValueError:
        illisibles += 1
        continue
    d = par_lancement[kid]
    d["nom"], d["grille"] = nom, grille
    d[met] = v * MULT.get(unit, 1)

def cle(d):
    # "void <unnamed>::int8_gemv_kernel<4, 12, ...>(...)" -> "int8_gemv_kernel<4, 12>"
    n = d["nom"]
    n = n[5:] if n.startswith("void ") else n
    n = n.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    base = re.match(r"[\w:]+", n)
    base = base.group(0) if base else n[:40]
    m = re.match(r"[\w:]+<([^<>()]*)>", n)
    if m and "at::" not in base:
        base += "<" + ", ".join(x.strip() for x in m.group(1).split(",")[:2]) + ">"
    n = base[-60:]
    if "int8_gemv" in n:
        return f"{n} grille={d['grille']}"
    return n

agg = collections.defaultdict(lambda: collections.defaultdict(float))
for d in par_lancement.values():
    a = agg[cle(d)]
    a["n"] += 1
    for m in ("gpu__time_duration.sum", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum", "sm__inst_executed.sum",
              "sm__inst_executed_pipe_tensor.sum", "sm__inst_executed_pipe_fma.sum", "sm__inst_executed_pipe_lsu.sum",
              "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__cycles_elapsed.avg.per_second",
              "launch__grid_size", "sm__warps_active.avg.pct_of_peak_sustained_active",
              "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"):
        a[m] += d.get(m, 0.0)

# La plus grande grille int8_gemv = lm_head (N = vocabulaire).
int8 = [k for k in agg if "int8_gemv" in k]
if int8:
    def gx(k):
        m = re.search(r"grille=\((\d+)", k); return int(m.group(1)) if m else 0
    # lm_head : une seule fois par pas (les projections, une par couche).
    seuls = [k for k in int8 if agg[k]["n"] <= 1.5 * pas]
    if seuls:
        lm = max(seuls, key=gx)
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
    rd = a["dram__bytes_op_read.sum"] / pas
    inst = a["sm__inst_executed.sum"] / pas
    ipo = inst / rd if rd else float("inf")
    bw = (a["dram__bytes_op_read.sum"] + a["dram__bytes_op_write.sum"]) / a["gpu__time_duration.sum"] / 1e9 if a["gpu__time_duration.sum"] else 0
    n = a["n"]
    return (f"{k[:w]:{w}s} {n/pas:7.1f} {t:8.3f} {rd/1e9:7.3f} {bw:7.0f} {inst/1e9:8.3f} {ipo:9.2f} "
            f"{100*a['sm__inst_executed_pipe_tensor.sum']/max(inst*pas,1):5.1f} {100*a['sm__inst_executed_pipe_fma.sum']/max(inst*pas,1):5.1f} "
            f"{100*a['sm__inst_executed_pipe_lsu.sum']/max(inst*pas,1):5.1f} {a['sm__throughput.avg.pct_of_peak_sustained_elapsed']/n:5.1f} "
            f"{a['sm__cycles_elapsed.avg.per_second']/max(n,1)/1e6:5.0f}"
            + (f" grille={a['launch__grid_size']/max(n,1):7.0f} warps_actifs={a['sm__warps_active.avg.pct_of_peak_sustained_active']/max(n,1):5.1f}%"
               + (f" SM_actifs={a['sm__cycles_active.avg.pct_of_peak_sustained_elapsed']/max(n,1):5.1f}%" if a.get("sm__cycles_active.avg.pct_of_peak_sustained_elapsed") else "")
               if a.get("launch__grid_size") else "")
            + (f" Go_ecr={a['dram__bytes_op_write.sum']/pas/1e9:6.3f}" if a.get("dram__bytes_op_write.sum") else ""))

W = 64
ent = f"{'':{W}s} {'appels':>7s} {'ms/pas':>8s} {'Go lus':>7s} {'Go/s':>7s} {'Ginst':>8s} {'inst/oct':>9s} {'tens%':>5s} {'fma%':>5s} {'lsu%':>5s} {'SM%':>5s} {'MHz':>5s}"
print(f"Par pas (moyenne sur {pas} pas), {len(par_lancement)} lancements profilés, {illisibles} valeurs illisibles")
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
