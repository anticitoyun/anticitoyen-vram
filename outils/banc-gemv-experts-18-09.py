"""Micro-banc de la GEMV groupée des experts (sage-lecture-profils-coder-17-09
§ 2 : « ≥ 85 % de bande ») — Coder-30B-A3B b = 12 : 128 experts, top_k 8,
K 2048, I 768 ; gate/up fusionnés + down, v1 (une passe de poids par paire
expert-jeton) contre v2 (paires triées par expert, poids lus une fois pour
≤ 4 jetons du même expert), en rejeu de graphe, sur 20 routages aléatoires
(la répétition d'experts varie : ~69 distincts pour 96 paires).

Octets comptés = experts DISTINCTS × (gate + up + down) — ce que le bus doit
lire au minimum ; le taux v1 est donc pénalisé par ses relectures, c'est le
point. Scellé (Sage) : experts 6,6 → ≤ 5,3 ms par pas (48 couches) ; ici
par couche : v1 attendu ≈ 0,137 ms (1,2 To/s), v2 ≤ 0,110 ms (≥ 1,52 To/s
= 85 %). Chaque bras est jugé identique au bit à v1.

    outils/carte.sh python outils/banc-gemv-experts-18-09.py [--rapide]
    ACVRAM_GROUPED_RPW=2 outils/carte.sh python outils/banc-gemv-experts-18-09.py   # un processus par valeur

Campagne RPW (sage-gemv-experts-rpw-18-09) : rpw = 2, puis 4, puis le témoin
rpw = 1 en fin ; scellé unique min(rpw = 2, 4) ≤ 6,7 ms/pas (v1), bit-exact
exigé ; faux ⇒ une passe ncu bornée avant toute ligne de noyau. La ligne de
régime est imprimée en tête et copiée dans le JSON (un fichier par valeur).
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.kernels import get_extension                                    # noqa: E402
from acvram.quant.nvfp4 import quantize_nvfp4                                # noqa: E402

REPET = 20 if "--rapide" not in sys.argv else 6
ROUTAGES = 20 if "--rapide" not in sys.argv else 5
E, TOPK, B, K, I, COUCHES = 128, 8, 12, 2048, 768, 48
BANDE = 1.79e12


def chrono(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPET):
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); g.replay(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a))
    return statistics.median(ts)


def pile(M, Kk, seed):
    g = torch.Generator().manual_seed(seed)
    ts = [quantize_nvfp4((torch.randn(M, Kk, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous().cuda()
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous().cuda()
    gs = torch.stack([t.global_scale.float().reshape(()) for t in ts]).contiguous().cuda()
    return qw, bs, gs


def main():
    from acvram.regime import regime_ligne
    ext = get_extension()
    assert ext is not None and hasattr(ext, "nvfp4_gemv_grouped_gateup_v2"), "extension sans v2"
    # ACVRAM_GROUPED_RPW est lu par le .cu au PREMIER lancement (static) : une
    # valeur par processus ; la ligne de régime en tête la nomme (Sage)
    rpw = int(os.environ.get("ACVRAM_GROUPED_RPW", "4"))
    ligne = regime_ligne()
    print(ligne, flush=True)
    print(f"ACVRAM_GROUPED_RPW={rpw} (une valeur par processus)", flush=True)
    qg, bg, gsg = pile(I, K, 1); qu, bu, gsu = pile(I, K, 2); qd, bd, gsd = pile(K, I, 3)
    octets_expert = (qg.shape[1] * qg.shape[2] + bg.shape[1] * bg.shape[2]) * 2 + qd.shape[1] * qd.shape[2] + bd.shape[1] * bd.shape[2]
    g = torch.Generator().manual_seed(18)
    res = []
    for r in range(ROUTAGES):
        topi = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(B)])
        eid = topi.reshape(-1).to(torch.int32).cuda()
        tok = torch.arange(B, dtype=torch.int32).repeat_interleave(TOPK).cuda()
        distincts = int(torch.unique(eid).numel())
        x = torch.randn(B, K, generator=g).to(torch.bfloat16).cuda()
        seq = torch.arange(B * TOPK, dtype=torch.int32).cuda()
        ordre = torch.argsort(eid, stable=True).to(torch.int32)
        eid_s, tok_s, seq_s = eid[ordre.long()].contiguous(), tok[ordre.long()].contiguous(), ordre.contiguous()

        def v1():
            act = ext.nvfp4_gemv_grouped_gateup(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)
            return ext.nvfp4_gemv_grouped(qd, bd, gsd, eid, seq, act, I)

        def v2():
            act = ext.nvfp4_gemv_grouped_gateup_v2(qg, bg, gsg, qu, bu, gsu, eid_s, tok_s, ordre, x, K, 0)
            return ext.nvfp4_gemv_grouped_v2(qd, bd, gsd, eid_s, seq_s, ordre, act, I)

        def xreg():
            act = ext.nvfp4_gemv_grouped_gateup_xreg(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)
            return ext.nvfp4_gemv_grouped_xreg(qd, bd, gsd, eid, seq, act, I)

        def v2_gateup_seul():
            return ext.nvfp4_gemv_grouped_gateup_v2(qg, bg, gsg, qu, bu, gsu, eid_s, tok_s, ordre, x, K, 0)

        def v1_gateup_seul():
            return ext.nvfp4_gemv_grouped_gateup(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)

        y1, y2 = v1(), v2()
        identique = bool(torch.equal(y1, y2))
        ms1, ms2 = chrono(v1), chrono(v2)
        a_xreg = hasattr(ext, "nvfp4_gemv_grouped_xreg")
        ms3 = chrono(xreg) if a_xreg else float("nan")
        identique_xreg = bool(torch.equal(y1, xreg())) if a_xreg else None
        ms1g, ms2g = chrono(v1_gateup_seul), chrono(v2_gateup_seul)
        octets = distincts * octets_expert
        res.append({"routage": r, "distincts": distincts, "octets": octets, "identique": identique,
                    "v1_ms": ms1, "v2_ms": ms2, "v1_To_s": octets / ms1 / 1e9, "v2_To_s": octets / ms2 / 1e9,
                    "v1_gateup_ms": ms1g, "v2_gateup_ms": ms2g,
                    "xreg_ms": ms3, "xreg_To_s": octets / ms3 / 1e9 if ms3 == ms3 else float("nan"),
                    "xreg_identique": identique_xreg})
        print(f"routage {r:2d} distincts={distincts:3d} {octets / 1e6:6.1f} Mo  v1 {ms1:.4f} ms ({octets / ms1 / 1e9:.2f} To/s)"
              f"  v2 {ms2:.4f} ms ({octets / ms2 / 1e9:.2f} To/s)  xreg {ms3:.4f} ms  gateup v1/v2 {ms1g:.4f}/{ms2g:.4f}"
              f"  identique v2={identique} xreg={identique_xreg}", flush=True)
    med = lambda k: statistics.median(r[k] for r in res)
    v1_pas, v2_pas = med("v1_ms") * COUCHES, med("v2_ms") * COUCHES
    tous = all(r["identique"] for r in res)
    verdict = ("OUVRE (v2 ≤ 5,3 ms par pas de 48 couches et identique au bit)" if v2_pas <= 5.3 and tous
               else "FAUX (v2 > 5,3 ms par pas)" if tous else "SORTIE DIFFÉRENTE : ne compte pas")
    # campagne RPW (sage-gemv-experts-rpw-18-09) : scellé sur v1 seule,
    # min(rpw = 2, 4) ≤ 6,7 ms/pas, bit-exact exigé, témoin rpw = 1 en fin
    verdict_rpw = (f"rpw={rpw} : v1 {v1_pas:.2f} ms/pas "
                   + ("(≤ 6,7 : tenu si bit-exact)" if v1_pas <= 6.7 else "(> 6,7)")
                   + (" — bit-exact v2/v1 sur tous les routages" if tous else " — SORTIE DIFFÉRENTE"))
    print(verdict_rpw)
    if all(r["xreg_ms"] == r["xreg_ms"] for r in res):
        xreg_pas = med("xreg_ms") * COUCHES
        tous_x = all(r["xreg_identique"] for r in res)
        # dernier geste (sage-gemv-experts-dernier-geste-18-09) : porte ncu gateup+down
        # ≤ 6,2 ms/pas ; ici le rejeu de graphe en donne l'ordre de grandeur
        print(f"xreg : {med('xreg_ms'):.4f} ms/couche ({med('xreg_To_s'):.2f} To/s) → {xreg_pas:.2f} ms/pas "
              f"(v1 rpw={rpw} : {v1_pas:.2f}) ; identique au bit {sum(bool(r['xreg_identique']) for r in res)}/{len(res)}"
              + (" — ≤ 6,2 : à confirmer sous ncu" if xreg_pas <= 6.2 else " — > 6,2"))
    print(f"\nmédianes : v1 {med('v1_ms'):.4f} ms/couche ({med('v1_To_s'):.2f} To/s, {med('v1_To_s') / 1.79 * 100:.0f} %) → {v1_pas:.2f} ms/pas ; "
          f"v2 {med('v2_ms'):.4f} ms/couche ({med('v2_To_s'):.2f} To/s, {med('v2_To_s') / 1.79 * 100:.0f} %) → {v2_pas:.2f} ms/pas ; "
          f"identique au bit sur {sum(r['identique'] for r in res)}/{len(res)} routages → {verdict}")
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", f"banc-gemv-experts-18-09-rpw{rpw}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0), "repet": REPET,
               "regime": ligne, "ACVRAM_GROUPED_RPW": rpw, "verdict_rpw": verdict_rpw,
               "xreg_ms_pas": med("xreg_ms") * COUCHES if all(r["xreg_ms"] == r["xreg_ms"] for r in res) else None,
               "formes": {"E": E, "top_k": TOPK, "b": B, "K": K, "I": I, "couches": COUCHES},
               "resultats": res, "v1_ms_pas": v1_pas, "v2_ms_pas": v2_pas, "verdict": verdict},
              open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
