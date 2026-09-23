#!/usr/bin/env python3
"""Pièce 76 : familles de noyaux de DEUX moteurs (acvram, vLLM) sur une taxonomie commune, en ms/pas de décodage,
durées de noyaux seules (les parts hôte sous nsys ne valent rien : le graphe d'acvram les masque).

Un pas = `couches` marqueurs consécutifs (un par couche : `_route_fusee_kernel` chez nous, `topkGating` chez vLLM) ;
on garde les fenêtres de décodage établi (mur ≤ 1,3 × la médiane des murs, hors 2 premières et dernière), puis on
somme par famille et on divise par le nombre de fenêtres gardées.

Usage : familles-comparees.py --acvram A.csv --vllm V.csv [--horloge-acvram MHz] [--horloge-vllm MHz] [--json S] [--detail N]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

MARQUEURS = {"acvram": r"_route_fusee_kernel", "vllm": r"moe::topkGating"}
# Ordre significatif : la première famille qui correspond l'emporte. La tête et le routeur partagent un GEMM
# cutlass chez vLLM et le noyau étroit int8 chez nous : ils sont séparés par la durée (la tête lit 311 Mo en bf16,
# 156 Mo en int8 : elle dure > 60 µs, un routeur < 15 µs).
FAMILLES = [
    ("moe_gemm", r"marlin_moe_wna16::Marlin|nvfp4_gemv_marlin|nvfp4_gemv_grouped|gemm_grouped_mma"),
    ("moe_glue", r"topkGating|moe_align_block_size|count_and_sort_expert|moe_sum|moe_reduce|moe_act|moe_aligner|"
                 r"moe_pack|_colle_moe|moe_route_pack|_route_fusee_kernel|route_logits_fusee|act_and_mul_kernel"),
    ("normes", r"fused_add_rms_norm|rms_norm|rmsnorm|add_norm|layer_norm"),   # avant proj_dense : un « …rms_norm_marlin_gemm » triton de vLLM est une norme
    ("attention", r"unified_attention|reduce_segments|_partiel|_reduce_kernel\b|flash_fwd|paged_attention|decode_attention|attn_"),
    ("proj_dense", r"marlin::Marlin<|_etroit|int8_gemv_kernel|int8_dequant_kernel|etroit_triton|gemvx::kernel|marlin_gemm|nvfp4_gemm|nvfp4_dense"),
    ("gemm_cutlass", r"cutlass|cublasLt|splitKreduce|wmma|gemm"),
    ("rope_kv", r"rope|reshape_and_cache|kv_write"),
    ("echantillonnage", r"sample|gumbel|argmax|topk_topp|sampler"),
    ("copies", r"CUDA memcpy|CUDA memset"),
    ("glue_torch", r"at::native::|at_cuda_detail|elementwise_kernel|direct_copy|CatArrayBatchedCopy|arange"),
]
SEUIL_TETE_NS = 60_000


def famille_de(nom: str, duree: int) -> str:
    for f, motif in FAMILLES:
        if re.search(motif, nom):
            if f in ("proj_dense", "gemm_cutlass") and duree > SEUIL_TETE_NS:
                return "tete"
            return "routeur_gemm" if f == "gemm_cutlass" else f
    return "autres"


def lire(chemin: str):
    with open(chemin) as f:
        r = csv.DictReader(f)
        nom_col = "Name" if "Name" in r.fieldnames else "Kernel Name"
        for row in r:
            yield int(float(row["Start (ns)"])), int(float(row["Duration (ns)"])), row[nom_col]


def decomposer(chemin: str, moteur: str, couches: int = 48, detail: int = 0) -> dict:
    noyaux = sorted(lire(chemin))
    marq = [t for t, _, n in noyaux if re.search(MARQUEURS[moteur], n)]
    bornes = [(marq[i], marq[i + couches]) for i in range(0, len(marq) - couches, couches)]
    murs = [b - a for a, b in bornes]
    med = statistics.median(murs)
    gardees = [ab for j, ab in enumerate(bornes) if 2 <= j < len(bornes) - 1 and murs[j] <= 1.3 * med]
    fam = defaultdict(lambda: [0, 0])
    par_nom = defaultdict(lambda: [0, 0])
    k, i = 0, 0
    for a, b in gardees:                                  # fenêtres disjointes et croissantes : un seul balayage
        while i < len(noyaux) and noyaux[i][0] < a:
            i += 1
        j = i
        while j < len(noyaux) and noyaux[j][0] < b:
            t, d, n = noyaux[j]
            f = famille_de(n, d)
            fam[f][0] += d; fam[f][1] += 1
            par_nom[(f, n[:80])][0] += d; par_nom[(f, n[:80])][1] += 1
            j += 1
        k += 1
    n = len(gardees)
    res = {"fenetres_gardees": n, "fenetres_totales": len(bornes),
           "mur_ms_pas": round(statistics.median([b - a for a, b in gardees]) / 1e6, 3),
           "noyaux_ms_pas": round(sum(v[0] for v in fam.values()) / n / 1e6, 3),
           "familles": {f: {"ms_pas": round(v[0] / n / 1e6, 3), "lancements_pas": round(v[1] / n, 1)}
                        for f, v in sorted(fam.items(), key=lambda kv: -kv[1][0])}}
    if detail:
        res["detail"] = [{"famille": f, "noyau": nm, "ms_pas": round(v[0] / n / 1e6, 3), "lancements_pas": round(v[1] / n, 1),
                          "us": round(v[0] / v[1] / 1e3, 2)} for (f, nm), v in sorted(par_nom.items(), key=lambda kv: -kv[1][0])[:detail]]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acvram", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--horloge-acvram", type=float)
    ap.add_argument("--horloge-vllm", type=float)
    ap.add_argument("--detail", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    A, V = decomposer(a.acvram, "acvram", detail=a.detail), decomposer(a.vllm, "vllm", detail=a.detail)
    print(f"{'famille':16s} {'acvram ms/pas':>14s} {'vLLM ms/pas':>12s} {'écart':>8s}   lancements A / V"
          f"   (horloge médiane : acvram {a.horloge_acvram} MHz, vLLM {a.horloge_vllm} MHz)")
    for f in sorted(set(A["familles"]) | set(V["familles"]),
                    key=lambda f: -(A["familles"].get(f, {}).get("ms_pas", 0) - V["familles"].get(f, {}).get("ms_pas", 0))):
        x, y = A["familles"].get(f, {"ms_pas": 0, "lancements_pas": 0}), V["familles"].get(f, {"ms_pas": 0, "lancements_pas": 0})
        print(f"{f:16s} {x['ms_pas']:14.3f} {y['ms_pas']:12.3f} {x['ms_pas'] - y['ms_pas']:+8.3f}   {x['lancements_pas']:6.1f} / {y['lancements_pas']:6.1f}")
    print(f"{'NOYAUX':16s} {A['noyaux_ms_pas']:14.3f} {V['noyaux_ms_pas']:12.3f} {A['noyaux_ms_pas'] - V['noyaux_ms_pas']:+8.3f}")
    print(f"{'mur':16s} {A['mur_ms_pas']:14.3f} {V['mur_ms_pas']:12.3f}   fenêtres {A['fenetres_gardees']}/{A['fenetres_totales']} · {V['fenetres_gardees']}/{V['fenetres_totales']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"acvram": A, "vllm": V, "horloge_acvram_mhz": a.horloge_acvram, "horloge_vllm_mhz": a.horloge_vllm},
                      f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
