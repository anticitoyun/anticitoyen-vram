"""I5 (30/09) : banc de la récurrence GDN du décodage — fla en place (servi) contre `gdn_tuiles` (J = 2, 4), et deux
planchers au même motif d'accès (tuiles [128, 8] d'un état [K, V] fp32) : copie en place (lecture + écriture, sans
calcul) et lecture seule. Un graphe = 48 lancements sur 48 états DISTINCTS (une couche chacun, comme un pas de
Qwen3.8) : l'état d'une couche (37,7 Mo à b=12) ne reste pas dans le L2 d'un rejeu à l'autre. Rend des µs par couche
(médiane de --rep rejeux) et l'égalité au bit fla/tuiles du premier pas (`torch.equal` sortie et état).

    ACVRAM_NOM=poste5-i5 ACVRAM_TYPE=mesure outils/carte.sh \\
        python outils/gpu/mesure/banc-gdn-recurrence.py --attendu <commit> --json i5.json
"""
import argparse, json, os, statistics, subprocess, sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from acvram.engine import gdn as G                                   # noqa: E402
from acvram.engine.gdn_tuiles import recurrence_tuiles               # noqa: E402

COUCHES, K, V, H = 48, 128, 128, 16


@triton.jit
def _copie_kernel(S, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    o_k, o_v = tl.arange(0, K), i_v * BV + tl.arange(0, BV)
    p = S + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p, tl.load(p) * 1.0000001)                            # écrit une valeur changée : pas de magasin supprimé


@triton.jit
def _lecture_kernel(S, out, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    o_k, o_v = tl.arange(0, K), i_v * BV + tl.arange(0, BV)
    p = S + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(out + i_nh * (V // BV) + i_v, tl.sum(tl.sum(tl.load(p), 0), 0))


def chrono(fns, rep):
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
    return statistics.median(ts), statistics.pstdev(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attendu", required=True, help="commit exigé de l'arbre (refus rc 65 sinon)")
    ap.add_argument("--lots", default="1,2,8,12,16"); ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--rep", type=int, default=200); ap.add_argument("--json")
    a = ap.parse_args()
    tete = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=os.path.dirname(__file__)).stdout.strip()
    if not tete.startswith(a.attendu):
        print(f"REFUS : HEAD {tete[:12]} ≠ attendu {a.attendu}", file=sys.stderr); sys.exit(65)
    dev = torch.device("cuda", 0)
    res = {"commit": tete, "hv": a.hv, "rep": a.rep, "lots": {}}
    for b in [int(x) for x in a.lots.split(",")]:
        gen = torch.Generator().manual_seed(b)
        r = lambda *s, e=1.0: (torch.randn(*s, generator=gen) * e).to(dev)
        q, k, v = r(b, 1, H, K), r(b, 1, H, K), r(b, 1, a.hv, V)
        gb, bb, A, dt = r(b, 1, a.hv), r(b, 1, a.hv), r(a.hv, e=0.5) - 1.0, r(a.hv, e=0.5)
        gg = torch.Generator(device=dev).manual_seed(b)              # 48 états tirés sur la carte (2,4 Go à b=16)
        etats = [torch.randn(b, a.hv, K, V, generator=gg, device=dev) * 0.1 for _ in range(COUCHES)]
        # au bit, premier pas, sur des copies
        S1, S2 = etats[0].clone(), etats[0].clone()
        o1 = G._recurrence_en_place(q, k, v, gb, bb, S1, A, dt)
        bits = {}
        for J in (2, 4):
            S2.copy_(etats[0]); o2 = recurrence_tuiles(q, k, v, gb, bb, S2, A, dt, J=J)
            bits[J] = bool(torch.equal(o1, o2) and torch.equal(S1, S2))
        out = torch.empty(b * a.hv * (V // 8), device=dev)
        bras = {
            "fla": [lambda S=S: G._recurrence_en_place(q, k, v, gb, bb, S, A, dt) for S in etats],
            "tuiles2": [lambda S=S: recurrence_tuiles(q, k, v, gb, bb, S, A, dt, J=2) for S in etats],
            "tuiles4": [lambda S=S: recurrence_tuiles(q, k, v, gb, bb, S, A, dt, J=4) for S in etats],
            "plancher_rw": [lambda S=S: _copie_kernel[(V // 8, b * a.hv)](S, K=K, V=V, BV=8, num_warps=1)
                            for S in etats],
            "plancher_r": [lambda S=S: _lecture_kernel[(V // 8, b * a.hv)](S, out, K=K, V=V, BV=8, num_warps=1)
                           for S in etats],
        }
        ligne = {"au_bit": bits, "octets_lus_par_couche": b * a.hv * K * V * 4}
        for nom in ("fla", "tuiles2", "tuiles4", "plancher_rw", "plancher_r", "fla"):   # fla deux fois : dérive
            m, s = chrono(bras[nom], a.rep)
            ligne.setdefault(nom, []).append({"us": round(m, 3), "sigma": round(s, 3)})
        res["lots"][b] = ligne
        f = ligne["fla"]
        print(f"b={b:2d} fla {f[0]['us']:.2f}/{f[1]['us']:.2f} µs ; tuiles2 {ligne['tuiles2'][0]['us']:.2f} ; "
              f"tuiles4 {ligne['tuiles4'][0]['us']:.2f} ; plancher r+w {ligne['plancher_rw'][0]['us']:.2f} ; "
              f"lecture {ligne['plancher_r'][0]['us']:.2f} ; au bit {bits}", flush=True)
        del etats; torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print("FIN banc-gdn-recurrence", flush=True)


if __name__ == "__main__":
    main()
