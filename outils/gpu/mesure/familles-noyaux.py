"""Décomposition du graphe de décodage par FAMILLE de noyaux, à sec, depuis
une trace nsys (`nsys stats --report cuda_gpu_trace --format csv`) : ms par
pas et lancements par pas, par famille (experts Marlin, projections étroites
int8, attention, routage, normes, rope/kv, tête, glue torch, copies).

Un pas = 48 `_route_fusee_kernel` consécutifs (un par couche) : la fenêtre
[route k·48, route (k+1)·48) est un cycle complet de couches décalé d une
demi-couche — en régime établi sa somme par famille est celle d un pas.
Les fenêtres de tête et de queue (chargement, prefill, chauffe) sont exclues
par `--exclure` (défaut : les 2 premières et la dernière).

Prise (poste2, ≤ 5 min) :
  nsys profile -t cuda --cuda-graph-trace=node -o graphe.nsys-rep \\
      outils/carte.sh python outils/gpu/mesure/frontiere-pas.py f.json 12 60
  nsys stats --report cuda_gpu_trace --format csv -o graphe graphe.nsys-rep
  python outils/gpu/mesure/familles-noyaux.py graphe_cuda_gpu_trace.csv [--json f.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict

FAMILLES = [
    ("experts_marlin", r"marlin_moe_wna16::Marlin|nvfp4_gemv_marlin|nvfp4_gemv_grouped|marlin_gemm|gemm_grouped_mma"),
    ("experts_quant_a4", r"nvfp4_quant_act|moe_act_quant|moe_route_pack_quant"),     # chemin mma-a4 seulement (absents du pas servi Marlin, 22/09)
    ("experts_glue", r"moe_reduce|moe_act|moe_pack|_colle_moe|moe_route_pack"),
    ("routage", r"_route_fusee_kernel|route_logits_fusee|moe_route_kernel|radix_sort|radixSort|Histogram|DeviceScan"),
    ("routeur_gemm", r"cutlass::Kernel2|cutlass_80_tensorop"),                       # le GEMM des logits du routeur (1/couche), pas la tête
    ("proj_etroites_int8", r"_etroit|int8_gemv_kernel|int8_dequant_kernel|etroit_triton|splitKreduce|gemvx::kernel"),
    ("attention", r"_partiel|_reduce_kernel\b|flash_fwd|paged_attention|decode_attention|attn_"),   # 22/09 : `_partiel_reduit_kernel` (compact) tombait dans `autres`
    ("normes", r"rmsnorm|add_norm|layer_norm"),
    ("rope_kv", r"rope_inplace_kernel|kv_write|rope_kernel"),
    ("tete", r"lm_head|tete_|wmma"),
    ("copies", r"CUDA memcpy|CUDA memset"),
    ("glue_torch", r"at::native::|at_cuda_detail|elementwise_kernel|direct_copy|CatArrayBatchedCopy|arange"),
]
MARQUEUR = "_route_fusee_kernel"


def famille_de(nom: str) -> str:
    for f, motif in FAMILLES:
        if re.search(motif, nom):
            return f
    return "autres"


def lire(chemin: str):
    with open(chemin) as f:
        r = csv.DictReader(f)
        nom_col = "Name" if "Name" in r.fieldnames else "Kernel Name"
        for row in r:
            yield int(float(row["Start (ns)"])), int(float(row["Duration (ns)"])), row[nom_col]


def decomposer(chemin: str, couches: int = 48, marqueur: str = MARQUEUR, exclure=(2, 1)) -> dict:
    noyaux = sorted(lire(chemin))
    bornes = [t for t, _, n in noyaux if marqueur in n][::couches]
    if len(bornes) < 2 + sum(exclure):
        raise SystemExit(f"{len(bornes)} pas trouvés ({marqueur} × {couches}) : trace trop courte")
    fenetres = list(zip(bornes[:-1], bornes[1:]))[exclure[0]: len(bornes) - 1 - exclure[1]]
    par_pas = []
    j = 0
    for d, fin in fenetres:
        ms, n = defaultdict(float), defaultdict(int)
        while j < len(noyaux) and noyaux[j][0] < d:
            j += 1
        k = j
        while k < len(noyaux) and noyaux[k][0] < fin:
            f = famille_de(noyaux[k][2])
            ms[f] += noyaux[k][1] / 1e6
            n[f] += 1
            k += 1
        par_pas.append((dict(ms), dict(n), (fin - d) / 1e6))
    familles = sorted({f for ms, _, _ in par_pas for f in ms}, key=lambda f: -sum(ms.get(f, 0) for ms, _, _ in par_pas))

    def mediane(v):
        v = sorted(v)
        return v[len(v) // 2]
    table = {f: {"ms_pas": round(mediane([ms.get(f, 0.0) for ms, _, _ in par_pas]), 3),
                 "lancements_pas": mediane([n.get(f, 0) for _, n, _ in par_pas])} for f in familles}
    total = sum(t["ms_pas"] for t in table.values())
    for t in table.values():
        t["part"] = round(t["ms_pas"] / total, 3) if total else 0.0
    mur = mediane([m for _, _, m in par_pas])
    return {"trace": chemin, "pas_juges": len(par_pas), "familles": table,
            "noyaux_ms_pas": round(total, 3), "mur_ms_pas": round(mur, 3),
            "trou_ms_pas": round(mur - total, 3), "lancements_pas": sum(t["lancements_pas"] for t in table.values())}


def detailler(chemin: str, famille: str, couches: int = 48, marqueur: str = MARQUEUR, exclure=(2, 1)) -> dict:
    """Une famille par noyau (nom court + grille GrdX×GrdY, quand la trace la
    porte) : ms/pas, lancements/pas, µs par lancement — pour situer chaque
    forme contre son plancher de bande (q, kv, o des étroites ne se jugent pas
    ensemble : 22/09, 1,335 ms/145 lancements sans savoir lesquels)."""
    with open(chemin) as f:
        r = csv.DictReader(f)
        nom_col = "Name" if "Name" in r.fieldnames else "Kernel Name"
        grille = all(c in r.fieldnames for c in ("GrdX", "GrdY"))
        noyaux = []
        for row in r:
            n = row[nom_col]
            if famille_de(n) != famille:
                continue
            court = re.sub(r"\(.*", "", n).replace("void ", "")[:60]
            if grille:
                court += f" [{row['GrdX']}x{row['GrdY']}]"
            noyaux.append((int(float(row["Start (ns)"])), int(float(row["Duration (ns)"])), court))
    bornes = [t for t, _, n in sorted(lire(chemin)) if marqueur in n][::couches]
    fenetres = list(zip(bornes[:-1], bornes[1:]))[exclure[0]: len(bornes) - 1 - exclure[1]]
    noyaux.sort()
    par_nom: dict[str, list[list[float]]] = defaultdict(lambda: [[0.0, 0] for _ in fenetres])
    for t, d, n in noyaux:
        for i, (a, b) in enumerate(fenetres):
            if a <= t < b:
                par_nom[n][i][0] += d / 1e6
                par_nom[n][i][1] += 1
                break
    out = {}
    for n, v in par_nom.items():
        ms = sorted(x[0] for x in v)[len(v) // 2]
        k = sorted(x[1] for x in v)[len(v) // 2]
        out[n] = {"ms_pas": round(ms, 3), "lancements_pas": k, "us_par_lancement": round(ms * 1e3 / k, 1) if k else 0.0}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["ms_pas"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--couches", type=int, default=48)
    ap.add_argument("--marqueur", default=MARQUEUR)
    ap.add_argument("--json")
    ap.add_argument("--detail", default="", help="famille à détailler par noyau (nom + grille) : ms/pas, lancements, µs par lancement")
    a = ap.parse_args()
    r = decomposer(a.trace, a.couches, a.marqueur)
    if a.detail:
        r["detail"] = detailler(a.trace, a.detail, a.couches, a.marqueur)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[familles] {r['pas_juges']} pas jugés · noyaux {r['noyaux_ms_pas']} ms/pas · mur {r['mur_ms_pas']} "
          f"· trou {r['trou_ms_pas']} · {r['lancements_pas']} lancements/pas")
    for f, t in r["familles"].items():
        print(f"  {f:20s} {t['ms_pas']:7.3f} ms  {t['part']:6.1%}  {t['lancements_pas']:5d} lancements")
    for nom, t in r.get("detail", {}).items():
        print(f"    {nom[:70]:70s} {t['ms_pas']:7.3f} ms  {t['lancements_pas']:4d} × {t['us_par_lancement']:6.1f} µs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
