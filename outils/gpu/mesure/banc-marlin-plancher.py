"""Marlin de décodage à b=12 : où sont les 6 % sous le plancher ? (Océane,
22/09, `oceane-ecart-trtllm-2` § 2 levier 2 — le seul poste au bit qui reste)

Le pas servi lit 5,42 Go d experts (D = 42 distincts par couche) en 3,94 ms
= 1,38 To/s, contre 1,55 « plancher » (`plancher-par-octets`) et 1,79 pic.
Ce banc isole le GEMV Marlin servi (`nvfp4_gemv_marlin_gateup` +
`nvfp4_gemv_marlin`, formes Coder E 128, K 2 048, I 768, top-8) sur des piles
synthétiques (mêmes octets), routage à D distincts contrôlé (`--D`, défaut
42 comme M2) ou réel (`--routages fichier.pt`), SOUS GRAPHE CUDA, et le
décompose :
  gateup, down       : chaque noyau seul en graphe (µs, To/s lus)
  couche             : les deux enchaînés (la dépendance réelle)
  8 couches          : 8 piles distinctes enchaînées — par couche, la part de
                       rampe (tête/queue de chaque noyau) qui s amortit ou pas
  l2_froid           : idem couche, avec un balayage de 256 Mo entre deux
                       rejeux (les échelles/tables ne restent pas en L2)
Où les 6 % peuvent être : (a) rampe et queue de vague par lancement
(≈ 2-3 µs × 2 noyaux × 48 = 0,2-0,3 ms : ce que `8 couches` ne réduit pas et
que le lancement programmatique (PDL) cacherait, au bit) ; (b) quantification
des vagues (blocs = D × tuiles N contre 170 SM) ; (c) L2 ; (d) horloge SM
sous plafond de puissance (Manon : `-lgc 2700` contre libre, hors banc).
Prédit (écrit avant) : gateup + down ≈ 1,45-1,55 To/s en graphe à 2 700 ;
`couche` = somme + 2-4 µs ; `8 couches` par couche = `couche` ± 1 µs (la
rampe ne s amortit pas : elle est dans la dépendance) ; `l2_froid` + ≤ 2 %.
Réfuté : `8 couches` par couche < `couche` − 3 µs (la rampe s amortit déjà :
PDL n apporterait rien) ; ou To/s ≥ 1,55 ici (le manque n est pas dans le
noyau mais dans le pas : régime d horloge/puissance, à Manon).

Usage : outils/carte.sh python outils/gpu/mesure/banc-marlin-plancher.py [--D 42] [--b 12]
        [--rep 200] [--routages f.pt] [--json SORTIE]        (≤ 3 min)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
import torch  # noqa: E402

E, TOPK, K, I = 128, 8, 2048, 768
OCTETS_EXPERT = (I * K // 2 + I * K // 16) * 2 + (K * I // 2 + K * I // 16)      # gate + up + down, E2M1 + E4M3
PLANCHER, PIC = 1.55e12, 1.79e12


def routage(b: int, D: int, gen: torch.Generator) -> torch.Tensor:
    """[b, top_k] d indices d experts : EXACTEMENT D experts distincts sur la
    couche (top_k ≤ D ≤ b·top_k), chacun au moins une fois, sans doublon par
    jeton — les paires réparties comme un routage réel (2,3 jetons par
    expert distinct à b = 12, D = 42)."""
    assert TOPK <= D <= min(E, b * TOPK)
    experts = torch.randperm(E, generator=gen)[:D].tolist()
    lignes: list[set] = [set() for _ in range(b)]
    ordre = list(range(b)) * TOPK
    ordre = [ordre[i] for i in torch.randperm(len(ordre), generator=gen).tolist()]
    for e in experts:                                  # chaque distinct au moins une fois
        for t in ordre:
            if len(lignes[t]) < TOPK and e not in lignes[t]:
                lignes[t].add(e); break
    for t in range(b):                                 # le reste : au hasard parmi les D
        while len(lignes[t]) < TOPK:
            lignes[t].add(experts[int(torch.randint(D, (1,), generator=gen))])
    topi = torch.tensor([[e for e in lignes[t]] for t in range(b)], dtype=torch.long)
    return topi[:, torch.randperm(TOPK, generator=gen)]


def distincts(topi: torch.Tensor) -> int:
    return int(torch.unique(topi).numel())


def piles(dev, graine: int):
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import quantize_nvfp4

    def pile(n, k, seed):
        g = torch.Generator().manual_seed(seed)
        ts = [quantize_nvfp4((torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(E)]
        qw = torch.stack([t.qweight for t in ts]).contiguous().to(dev)
        bs = torch.stack([t.block_scale for t in ts]).contiguous().to(dev)
        gs = torch.stack([t.global_scale.float().reshape(()) for t in ts]).contiguous().to(dev)
        return MP.preparer_pile(qw, bs, gs)
    return pile(I, K, graine), pile(I, K, graine + 1), pile(K, I, graine + 2)


def chrono_graphe(fn, rep: int, entre=None) -> list[float]:
    """µs par rejeu d un graphe capturant `fn` ; `entre` : fonction hôte
    exécutée entre deux rejeux (balayage L2), hors chrono."""
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
        if entre is not None:
            entre()
            torch.cuda.synchronize()
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); g.replay(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a) * 1e3)
    return ts


def resume(ts: list[float], octets: int) -> dict:
    ts = sorted(ts)
    med = ts[len(ts) // 2]
    return {"us_mediane": round(med, 1), "us_p10": round(ts[int(0.1 * (len(ts) - 1))], 1),
            "us_p90": round(ts[int(0.9 * (len(ts) - 1))], 1),
            "to_s": round(octets / (med * 1e-6) / 1e12, 3), "part_plancher": round(octets / (med * 1e-6) / PLANCHER, 3),
            "part_pic": round(octets / (med * 1e-6) / PIC, 3)}


def mesurer(a) -> dict:
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin_gateup"):
        sys.exit("ÉCHEC / CAUSE : extension sans nvfp4_gemv_marlin_gateup / SUITE : build")
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(a.graine)
    if a.routages:
        reels = [t for t in torch.load(a.routages) if t.shape[1] == TOPK and t.shape[0] >= a.b]
        topi = reels[0][: a.b].to(torch.long)
    else:
        topi = routage(a.b, a.D, gen)
    D = distincts(topi)
    G = a.b * TOPK
    eid = topi.reshape(-1).to(torch.int32).to(dev)
    tok = torch.arange(a.b, dtype=torch.int32).repeat_interleave(TOPK).to(dev)
    seq = torch.arange(G, dtype=torch.int32, device=dev)
    x = (torch.randn(a.b, K, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
    couches = [piles(dev, 100 + 3 * c) for c in range(8)]
    mg, mu, md = couches[0]
    octets_gu = D * (I * K // 2 + I * K // 16) * 2
    octets_d = D * (K * I // 2 + K * I // 16)
    act0 = ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, I, 0)

    def gateup():
        return ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, I, 0)

    def down():
        return ext.nvfp4_gemv_marlin(*md, eid, seq, act0, I, K)

    def couche(p=couches[0], xx=x):
        g_, u_, d_ = p
        act = ext.nvfp4_gemv_marlin_gateup(*g_, *u_, eid, tok, xx, K, I, 0)
        return ext.nvfp4_gemv_marlin(*d_, eid, seq, act, I, K)

    def huit():
        y = None
        for p in couches:
            y = couche(p, x)
        return y
    balai = torch.empty(256 * 2 ** 20, dtype=torch.uint8, device=dev)

    r = {"b": a.b, "D": D, "D_demande": a.D, "top_k": TOPK, "rep": a.rep, "routages": a.routages,
         "octets_couche": octets_gu + octets_d, "plancher_to_s": PLANCHER / 1e12, "pic_to_s": PIC / 1e12}
    r["gateup"] = resume(chrono_graphe(gateup, a.rep), octets_gu)
    r["down"] = resume(chrono_graphe(down, a.rep), octets_d)
    r["couche"] = resume(chrono_graphe(couche, a.rep), octets_gu + octets_d)
    h = chrono_graphe(huit, a.rep // 2)
    r["huit_couches_par_couche"] = resume([t / 8 for t in h], octets_gu + octets_d)
    r["l2_froid"] = resume(chrono_graphe(couche, a.rep // 2, entre=lambda: balai.fill_(1)), octets_gu + octets_d)
    somme = r["gateup"]["us_mediane"] + r["down"]["us_mediane"]
    r["rampe_dependance_us"] = round(r["couche"]["us_mediane"] - somme, 1)
    r["amortissement_8_us"] = round(r["couche"]["us_mediane"] - r["huit_couches_par_couche"]["us_mediane"], 1)
    r["l2_us"] = round(r["l2_froid"]["us_mediane"] - r["couche"]["us_mediane"], 1)
    r["pas_48_couches_ms"] = round(r["couche"]["us_mediane"] * 48 / 1e3, 3)
    r.update(verdict(r))
    return r


def verdict(r: dict) -> dict:
    to_s = r["couche"]["to_s"]
    if to_s >= PLANCHER / 1e12:
        v = "RÉFUTÉ (le noyau isolé est AU plancher : les 6 % sont dans le pas — horloge/puissance/voisinage, à Manon)"
    elif r["amortissement_8_us"] > 3.0:
        v = f"RÉFUTÉ (8 couches enchaînées gagnent {r['amortissement_8_us']} µs/couche : la rampe s amortit déjà, PDL n apporterait rien)"
    elif r["l2_us"] > 0.02 * r["couche"]["us_mediane"]:
        v = f"TENU, L2 : {r['l2_us']} µs/couche de plus à L2 froid (> 2 %) — le service est plus proche du froid que du chaud"
    else:
        v = (f"TENU : {to_s} To/s = {r['couche']['part_plancher']:.0%} du plancher ; rampe de dépendance {r['rampe_dependance_us']} µs/couche "
             f"× 48 = {r['rampe_dependance_us'] * 48 / 1e3:.2f} ms/pas — c est la part que le lancement programmatique (PDL) peut rendre, au bit")
    return {"verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--D", type=int, default=42)
    ap.add_argument("--b", type=int, default=12)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--routages")
    ap.add_argument("--graine", type=int, default=22)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = mesurer(a)
    r["part_plancher"] = r["couche"]["part_plancher"]
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[marlin-plancher] b={r['b']} D={r['D']} · {r['octets_couche'] / 1e6:.1f} Mo/couche · pas 48 couches {r['pas_48_couches_ms']} ms")
    for k in ("gateup", "down", "couche", "huit_couches_par_couche", "l2_froid"):
        t = r[k]
        print(f"  {k:26s} {t['us_mediane']:7.1f} µs (p10 {t['us_p10']:.1f} p90 {t['us_p90']:.1f})  {t['to_s']:.3f} To/s  {t['part_plancher']:.0%} plancher  {t['part_pic']:.0%} pic")
    print(f"  rampe de dépendance {r['rampe_dependance_us']} µs/couche · amortissement 8 couches {r['amortissement_8_us']} µs · L2 froid +{r['l2_us']} µs")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
