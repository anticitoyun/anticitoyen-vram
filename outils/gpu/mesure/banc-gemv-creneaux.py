#!/usr/bin/env python3
"""Pièce 61 (1) : banc du noyau seul — GEMV Marlin par paire (servi) contre par créneau d'expert
(`nvfp4_gemv_marlin_gateup_slots` / `nvfp4_gemv_marlin_slots`, TPB 4 et 8), formes réelles du Coder
(gate·up K 2048 → N 768, down K 768 → N 2048, E 128, k 8), godet 16 avec 12 jetons réels et 4 fantômes
(G = 128 paires), routage tiré pour donner ≈ 33 experts distincts par couche (pièce 60), L2 froid
(COPIES piles d experts en rotation dans un graphe), 200 répétitions. Publie µs par couche (gate·up,
down, moe_slots), ms/pas équivalent × 48 couches, et l égalité AU BIT (torch.equal) des sorties [G, N]
contre le noyau par paire — fantômes compris.

    outils/carte.sh python outils/gpu/mesure/banc-gemv-creneaux.py --json creneaux.json
"""
import argparse, json, os, statistics, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from acvram import kernels

E, K_GU, N_GU, K_D, N_D, TOPK, B, GODET = 128, 2048, 768, 768, 2048, 8, 12, 16
COPIES = int(os.environ.get("BANC_COPIES", "8"))
COUCHES = 48


def pile(k, n, graine, dev):
    g = torch.Generator().manual_seed(graine)
    w = torch.randint(-2**31, 2**31 - 1, (E, k // 16, n * 2), generator=g, dtype=torch.int32).to(dev)
    s = torch.randint(0x30, 0x40, (E, k // 16, n), generator=g, dtype=torch.uint8).to(dev)          # e5m3 : exposants modérés
    gs = torch.full((E,), 2.0 ** 119 * 1e-3, dtype=torch.float32, device=dev)
    return w, s, gs


def routage(graine, dev, concentre=True):
    """Paires (expert, jeton) d un godet 16 : 12 jetons × 8 experts distincts tirés sur une loi
    concentrée (≈ 33 experts distincts par couche, comme mesuré), 4 lignes fantômes (e = -1)."""
    g = torch.Generator().manual_seed(graine)
    p = (1.0 / (torch.arange(E, dtype=torch.float32) + 1) ** 1.1) if concentre else torch.ones(E)
    eid = torch.full((GODET * TOPK,), -1, dtype=torch.int32)
    tok = torch.zeros(GODET * TOPK, dtype=torch.int32)
    for t in range(B):
        ex = torch.multinomial(p, TOPK, replacement=False, generator=g)
        eid[t * TOPK:(t + 1) * TOPK] = ex.to(torch.int32); tok[t * TOPK:(t + 1) * TOPK] = t
    return eid.to(dev), tok.to(dev)


def routage_extremes(dev):
    """Cas limites du test au bit : un expert qui reçoit les 12 jetons, d autres à 1 jeton, fantômes."""
    eid = torch.full((GODET * TOPK,), -1, dtype=torch.int32); tok = torch.zeros(GODET * TOPK, dtype=torch.int32)
    for t in range(B):
        ex = [7] + [(t * 13 + i * 17) % E for i in range(1, TOPK)]            # expert 7 pour tous, les autres dispersés
        for i, e in enumerate(ex): eid[t * TOPK + i] = e; tok[t * TOPK + i] = t
    return eid.to(dev), tok.to(dev)


def chrono_graphe(fns, rep):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3 / len(fns))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep", type=int, default=200); ap.add_argument("--json")
    a = ap.parse_args()
    ext = kernels.get_extension()
    for nom in ("moe_slots", "nvfp4_gemv_marlin_slots", "nvfp4_gemv_marlin_gateup_slots"):
        assert hasattr(ext, nom), f"extension sans {nom} : recompiler"
    dev = torch.device("cuda", 0)
    piles = [(pile(K_GU, N_GU, 10 + i, dev), pile(K_GU, N_GU, 100 + i, dev), pile(K_D, N_D, 200 + i, dev)) for i in range(COPIES)]
    x = (torch.randn(GODET, K_GU, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16).to(dev)
    routages = [routage(1000 + i, dev) for i in range(COPIES)]
    distincts = [int(torch.unique(e[e >= 0]).numel()) for e, _ in routages]
    r = {"E": E, "godet": GODET, "b": B, "copies": COPIES, "rep": a.rep, "experts_distincts_par_couche": distincts, "bras": {}}

    def paire(i):
        (wg, sg, gg), (wu, su, gu), (wd, sd, gd) = piles[i]; eid, tok = routages[i]
        act = ext.nvfp4_gemv_marlin_gateup(wg, sg, gg, wu, su, gu, eid, tok, x, K_GU, N_GU, 0, None)
        seq = torch.arange(eid.shape[0], device=dev, dtype=torch.int32)
        return act, ext.nvfp4_gemv_marlin(wd, sd, gd, eid, seq, act.contiguous(), K_D, N_D, None)

    def creneau(i, tpb):
        (wg, sg, gg), (wu, su, gu), (wd, sd, gd) = piles[i]; eid, tok = routages[i]
        se, sp = ext.moe_slots(eid, E, tpb)
        act = ext.nvfp4_gemv_marlin_gateup_slots(wg, sg, gg, wu, su, gu, se, sp, tok, x, K_GU, N_GU, 0, tpb, None)
        seq = torch.arange(eid.shape[0], device=dev, dtype=torch.int32)
        return act, ext.nvfp4_gemv_marlin_slots(wd, sd, gd, se, sp, seq, act.contiguous(), K_D, N_D, tpb, None)

    # 1. au bit, routage concentré et cas extrêmes
    egal = {}
    for tpb in (4, 8):
        ok = True; ecart = 0.0
        for i in range(COPIES):
            a1, d1 = paire(i); a2, d2 = creneau(i, tpb)
            ok = ok and torch.equal(a1, a2) and torch.equal(d1, d2)
            ecart = max(ecart, float((a1 - a2).abs().max()), float((d1 - d2).abs().max()))
        eid, tok = routage_extremes(dev); routages_sauve = routages[0]; routages[0] = (eid, tok)
        a1, d1 = paire(0); a2, d2 = creneau(0, tpb); routages[0] = routages_sauve
        ok_ext = torch.equal(a1, a2) and torch.equal(d1, d2)
        fant = bool((a2[eid < 0] == 0).all() and (d2[eid < 0] == 0).all())
        egal[tpb] = {"au_bit": bool(ok), "ecart_max": ecart, "au_bit_extremes": bool(ok_ext), "fantomes_a_zero": fant}
    r["au_bit"] = egal
    # 2. chrono L2 froid (rotation des COPIES piles) : µs par couche
    t_paire = chrono_graphe([(lambda i=i: paire(i)) for i in range(COPIES)], a.rep)
    r["bras"]["paire (servi)"] = {"us_couche": round(t_paire, 2), "ms_pas_equiv": round(t_paire * COUCHES / 1e3, 3)}
    for tpb in (4, 8):
        t = chrono_graphe([(lambda i=i, tpb=tpb: creneau(i, tpb)) for i in range(COPIES)], a.rep)
        t_slots = chrono_graphe([(lambda i=i, tpb=tpb: ext.moe_slots(routages[i][0], E, tpb)) for i in range(COPIES)], a.rep)
        r["bras"][f"créneaux TPB{tpb}"] = {"us_couche": round(t, 2), "us_moe_slots": round(t_slots, 2), "ms_pas_equiv": round(t * COUCHES / 1e3, 3),
                                          "gain_pct": round(100 * (t_paire - t) / t_paire, 1)}
    print("RESULTAT " + json.dumps(r, ensure_ascii=False), flush=True)
    if a.json: json.dump(r, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
