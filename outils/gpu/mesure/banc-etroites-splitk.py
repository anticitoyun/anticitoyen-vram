"""Projections étroites int8 (q/kv, o) à b=12 : balayage du nombre de
tranches split-K de `_etroit_reduit_kernel` (poste1, 22/09, conception
`poste1-etroites-splitk-conception-22-09`). En service : qkv [5120 × 2048]
= 80 tuiles × 4 tranches, 10,3 µs = 1,02 To/s ; o [2048 × 4096] = 32 × 11,
11,8 µs = 0,71 To/s (verdict-detail-etroites-22-09) — contre 1,55 possible.

Pour chaque forme (formes de Coder, poids int8 synthétiques de même taille,
x [12, K] bf16), pour chaque nombre de tranches T du balayage : temps sous
graphe CUDA (µs médian, p90), To/s, et **écart au chemin EXACT** (T = auto,
celui que le défaut sert) : max |Δ| en ulp bf16 de |y|, part des éléments
qui diffèrent, max |Δ| relatif fp32 — c est la borne « ± 1 ulp » de
REGLES § 1 que l opt-in doit tenir, mesurée sur des entrées réelles de
forme. Le kernel est celui du service (mêmes constantes BM/BN/warps) :
seul T change (`ACVRAM_ETROITES_SPLITK`), donc seule la partition de K
change — l ordre fp32 des tranches sommées par le dernier programme.

Prédit (écrit avant) : l optimum est à T ≈ 2-3 programmes par SM (qkv 8-11,
o 16-22) : qkv ≤ 7,5 µs, o ≤ 6,5 µs (≥ 1,3 To/s) ; écart ≤ 1 ulp bf16 sur
< 5 % des éléments, 0 ulp sinon. Réfuté si aucun T ne gagne ≥ 2 µs sur les
deux formes (la marge n est pas dans les tranches : occupation par
programme, à `ptxas`), ou si un T gagnant dépasse 1 ulp (alors l opt-in ne
peut pas s appeler « ± 1 ulp »).

Usage : outils/carte.sh python outils/gpu/mesure/banc-etroites-splitk.py [--tranches 2,4,6,8,11,16,22,32]
        [--rep 200] [--json SORTIE]                                     (≤ 1 min)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

FORMES = {"qkv": (5120, 2048), "o": (2048, 4096)}       # Coder-30B : q 4096 + k 512 + v 512 ; o
B, G = 12, 128
PLANCHER = 1.55e12


def ulp_bf16(y: torch.Tensor) -> torch.Tensor:
    """L ulp bf16 de chaque élément (2^(exposant − 7)), pour mesurer |Δ| en ulp de |y|."""
    a = y.abs().to(torch.float32).clamp(min=2.0 ** -126)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def ecart(y: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (y.to(torch.float32) - ref.to(torch.float32)).abs()
    en_ulp = d / ulp_bf16(ref)
    rel = (d / ref.abs().to(torch.float32).clamp(min=1e-6)).max().item()
    return {"ulp_max": round(en_ulp.max().item(), 3), "part_differents": round((d > 0).float().mean().item(), 5),
            "rel_max": rel}


def poids_int8(n: int, k: int, graine: int, device):
    from acvram.quant.formats import quantize
    g = torch.Generator().manual_seed(graine)
    w = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
    t = quantize(w, "int8", group_size=G)
    return t.to_device(device) if hasattr(t, "to_device") else t


def chrono(fn, rep: int) -> list[float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return ts


def mesurer(tranches: list[int], rep: int) -> dict:
    from acvram.kernels import gemm_etroit as GE
    dev = torch.device("cuda")
    r = {"tranches_balayees": tranches, "rep": rep, "b": B, "formes": {}}
    for nom, (n, k) in FORMES.items():
        t = poids_int8(n, k, graine=hash(nom) % 1000, device=dev)
        x = (torch.randn(B, k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
        octets = n * k + t.scales.numel() * 2 + t.zeros.numel()
        # chemin exact = défaut (T auto) : la référence de l écart
        GE.regler_tranches(None)
        ref = GE.gemm_etroit(x, t, compact=True).clone()
        t_auto = GE.tranches_de(x, t)
        lignes = {}
        for T in [None] + [T for T in tranches if T != t_auto]:
            GE.regler_tranches(T)
            y = GE.gemm_etroit(x, t, compact=True)
            ts = sorted(chrono(lambda: GE.gemm_etroit(x, t, compact=True), rep))
            med = ts[len(ts) // 2]
            lignes[str(T if T is not None else f"auto={t_auto}")] = {
                "us": round(med, 2), "us_p90": round(ts[int(0.9 * (len(ts) - 1))], 2),
                "to_s": round(octets / (med * 1e-6) / 1e12, 3), "part_plancher": round(octets / (med * 1e-6) / PLANCHER, 3),
                "programmes": GE.programmes_de(x, t), **ecart(y, ref)}
        GE.regler_tranches(None)
        r["formes"][nom] = {"forme": [n, k], "octets": octets, "auto": t_auto, "lignes": lignes}
    r.update(verdict(r))
    return r


def verdict(r: dict) -> dict:
    gains, ulps = {}, {}
    for nom, f in r["formes"].items():
        base = next(v for k, v in f["lignes"].items() if k.startswith("auto"))
        cands = [(k, v) for k, v in f["lignes"].items() if not k.startswith("auto") and v["ulp_max"] <= 1.0]
        if cands:
            k, v = min(cands, key=lambda kv: kv[1]["us"])
            gains[nom] = {"T": k, "us_gain": round(base["us"] - v["us"], 2), "to_s": v["to_s"], "ulp_max": v["ulp_max"],
                          "part_differents": v["part_differents"]}
        ulps[nom] = max((v["ulp_max"] for v in f["lignes"].values()), default=0.0)
    if not gains or all(g["us_gain"] < 2.0 for g in gains.values()):
        v = "RÉFUTÉ : aucun T à ≤ 1 ulp ne gagne ≥ 2 µs — la marge n est pas dans les tranches (occupation par programme : ptxas)"
    else:
        total = sum(max(g["us_gain"], 0.0) for g in gains.values()) * 48 / 1e3
        v = (f"TENU : " + " ; ".join(f"{n} T={g['T']} −{g['us_gain']} µs ({g['to_s']} To/s, {g['ulp_max']} ulp max, "
                                     f"{g['part_differents']:.2%} éléments ≠)" for n, g in gains.items())
             + f" → −{total:.2f} ms/pas")
    return {"gains": gains, "ulp_max_balayage": ulps, "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tranches", default="2,4,6,8,11,16,22,32")
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = mesurer([int(t) for t in a.tranches.split(",")], a.rep)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    for nom, f in r["formes"].items():
        print(f"[etroites] {nom} {f['forme']} · {f['octets'] / 1e6:.1f} Mo · auto = {f['auto']} tranches")
        for k, v in f["lignes"].items():
            print(f"    T={k:8s} {v['us']:6.2f} µs (p90 {v['us_p90']:.2f}) {v['to_s']:.3f} To/s {v['part_plancher']:.0%}  "
                  f"prog {v['programmes']:4d}  ulp max {v['ulp_max']:.2f}  ≠ {v['part_differents']:.2%}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
