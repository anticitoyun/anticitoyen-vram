"""Porte micro-banc du GEMM dense étroit NVFP4 (sage-hybrides-etape1-close-
gemm-dense-17-09 § 2) : octets de poids / temps, en rejeu de graphe, sur les
formes de Qwen3.8-27B à M = 12 (et 2, 32 en contrôle).

Palier 1 (Laure 21033b7) : min sur les formes 0,47 (kv) < 0,9 → porte
fermée ; le taux suit N. Palier 2 (sage-gemm-dense-porte-fermee-palier-17-09
§ 3) : multi-projection q/k/v et qkv/gate/α/β en un lancement, tranches K
plus nombreuses (programmes par SM 2/4/8), réduction fusionnée ; juge =
taux PONDÉRÉ PAR LES OCTETS du pas ≥ 1,0 To/s OUVRE, < 0,85 FAUX (noyau
CUDA, décision séparée). Témoins dans le même banc : la boucle GEMV actuelle
(`ext.nvfp4_gemv`, attendu ~0,23 To/s à M = 12) et `narrow_gemm`
(NARROW_NVFP4, attendu pire). Chaque bras est jugé exact contre la
déquantification (2⁻⁷ × Σ|x·w|) : un chiffre de bande sur une sortie fausse
ne compte pas.

    outils/carte.sh python outils/banc-gemm-dense-etroit-17-09.py [--rapide]

Sortie : tableau + JSON dans scratchpad/banc-gemm-dense-etroit-17-09.json.
"""
import itertools
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.kernels import gemm_dense_etroit as GD, get_extension            # noqa: E402
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4              # noqa: E402

REPET = 30 if "--rapide" not in sys.argv else 8
BANDE_CRETE = 1.79e12           # 5090, octets/s (plaque)
# (nom, N ou tailles d'une multi-projection, K, couches par pas) — Qwen3.8-27B :
# H 5120, I 17408, 24 têtes q × 256, 4 kv, 17 couches d'attention + 48 GDN
# (qkv 10240, gate 6144, α/β 48), 65 MLP. Palier 2 (sage-gemm-dense-porte-
# fermee-palier-17-09 § 3) : q/k/v en UN lancement (multi-projection, chacune
# avec son scaler — l'empilement est refusé sur calibA, act_scale q ≠ k ≠ v),
# idem qkv+gate+α+β du GDN ; le juge est le taux PONDÉRÉ PAR LES OCTETS du pas.
FORMES = [("q_proj", 6144, 5120, 0), ("kv_proj", 2048, 5120, 0),                # séparées : témoins du palier 1
          ("qkv_multi", (6144, 1024, 1024), 5120, 17), ("o_proj", 5120, 6144, 17),
          ("gdn_multi", (10240, 6144, 48, 48), 5120, 48), ("gdn_out", 5120, 6144, 48),
          ("gate_up", 34816, 5120, 65), ("down", 5120, 17408, 65)]
MS = [12] if "--rapide" in sys.argv else [2, 12, 32]
# (BN, BK, warps, stages, programmes par SM visés par les tranches K)
CONFIGS = [(64, 128, 4, 3, 2), (128, 128, 4, 3, 2), (64, 128, 4, 4, 2), (128, 256, 8, 3, 2),
           (64, 128, 4, 3, 4), (128, 128, 4, 3, 4), (64, 128, 4, 3, 8), (128, 128, 4, 3, 8)]
if "--rapide" in sys.argv:
    CONFIGS = CONFIGS[:4]


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


def juger(y, x, t):
    wd = dequantize_nvfp4(t, torch.float32)
    attendu = x.float() @ wd.T
    borne = x.float().abs() @ wd.abs().T
    return int(((y.float() - attendu).abs() > 2 ** -7 * borne).sum())


def main():
    ext = get_extension()
    assert ext is not None and GD.disponible()
    dev = torch.device("cuda")
    res = []
    g = torch.Generator().manual_seed(17)
    from acvram.engine.layers import QuantLinear
    from acvram.quant.calibrate import ChannelScaler
    for nom, N, K, couches in FORMES:
        multi = isinstance(N, tuple)
        tailles = N if multi else (N,)
        ts = []
        for n in tailles:
            w = (torch.randn(n, K, generator=g) * 0.02).to(torch.bfloat16)
            t = quantize_nvfp4(w)
            t.qweight, t.block_scale, t.global_scale = t.qweight.to(dev), t.block_scale.to(dev), t.global_scale.to(dev)
            ts.append(t)
        t = ts[0]
        N_tot = sum(tailles)
        octets = sum(t_.qweight.numel() + t_.block_scale.numel() for t_ in ts)
        # scalers distincts par projection (ce que calibA impose)
        lins = [QuantLinear(t_, None, scaler=ChannelScaler((0.5 + torch.rand(K, generator=g) * (1 + i)).to(torch.float16).to(dev), 0))
                for i, t_ in enumerate(ts)]
        mp = GD.MultiProjection(lins) if multi else None
        for M in MS:
            x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(dev)
            bras = {}
            if not multi:
                # témoin 1 : la boucle GEMV actuelle (nvfp4_matmul, n ≤ 32)
                qw, bs, gsf = t.qweight.contiguous(), t.block_scale.view(torch.uint8).contiguous(), t.global_scale_float()
                f = lambda: ext.nvfp4_gemv(qw, bs, gsf, x, t.padded_in, None)
                bras["gemv_boucle"] = (chrono(f), juger(f()[:, :N], x, t), "")
                # témoin 2 : narrow_gemm (NARROW_NVFP4)
                if hasattr(ext, "narrow_gemm") and M <= 16 and K % 64 == 0:
                    from acvram.kernels import _narrow_rows
                    f = lambda: ext.narrow_gemm(qw, bs, None, None, x, t.padded_in, 16, gsf, _narrow_rows(N))
                    try:
                        bras["narrow_gemm"] = (chrono(f), juger(f()[:, :N], x, t), "")
                    except Exception as exc:                          # noqa: BLE001
                        bras["narrow_gemm"] = (float("nan"), -1, type(exc).__name__)
            else:
                # témoin : les projections séparées, chacune par le noyau dense (palier 1)
                def sep():
                    return torch.cat([GD.gemm_dense_etroit(l.scaler.apply(x), l.qweight) for l in lins], 1)
                bras["separees_triton"] = (chrono(sep), sum(juger(y_, l.scaler.apply(x), l.qweight) for y_, l in
                                                            zip(torch.split(sep(), tailles, 1), lins)), "")
            # candidat : balayage des configurations
            for bn, bk, wp, st, pps in CONFIGS:
                GD._PROGRAMMES_PAR_SM = pps
                if multi:
                    f = lambda: mp(x, bn=bn, bk=bk, warps=wp, stages=st)
                    jug = lambda y_: sum(juger(y_p, l.scaler.apply(x), l.qweight)
                                         for y_p, l in zip(torch.split(y_, tailles, 1), lins))
                else:
                    f = lambda: GD.gemm_dense_etroit(x, t, bn=bn, bk=bk, warps=wp, stages=st)
                    jug = lambda y_: juger(y_, x, t)
                cle = f"triton_bn{bn}_bk{bk}_w{wp}_s{st}_p{pps}"
                try:
                    bras[cle] = (chrono(f), jug(f()), "")
                except Exception as exc:                          # noqa: BLE001
                    bras[cle] = (float("nan"), -1, type(exc).__name__)
            GD._PROGRAMMES_PAR_SM = 2
            for b, (ms, hors, err) in bras.items():
                tos = octets / (ms * 1e-3) / 1e12 if ms == ms else float("nan")
                res.append({"forme": nom, "N": N_tot, "tailles": list(tailles), "K": K, "M": M, "couches": couches,
                            "octets": octets, "bras": b, "ms": ms, "To_s": tos,
                            "part_crete": tos * 1e12 / BANDE_CRETE, "hors_2m7": hors, "erreur": err})
                print(f"{nom:9s} N={N_tot:5d} K={K:5d} M={M:2d} {b:30s} {ms:8.3f} ms {tos:5.2f} To/s "
                      f"({tos * 1e12 / BANDE_CRETE * 100:4.0f} %) hors={hors}{' ' + err if err else ''}", flush=True)
    # verdict : meilleure config Triton exacte par forme à M = 12, et le minimum sur les formes
    meilleurs = {}
    for r in res:
        if r["M"] == 12 and r["bras"].startswith("triton") and r["hors_2m7"] == 0:
            if r["forme"] not in meilleurs or r["To_s"] > meilleurs[r["forme"]]["To_s"]:
                meilleurs[r["forme"]] = r
    print("\nMeilleure configuration Triton exacte par forme (M = 12) :")
    for nom, r in meilleurs.items():
        print(f"  {nom:8s} {r['bras']:26s} {r['To_s']:.2f} To/s")
    mini = min((r["To_s"] for r in meilleurs.values()), default=float("nan"))
    # palier 2 : taux pondéré par les octets du PAS (couches × octets) sur les
    # formes servies (couches > 0), meilleure config exacte par forme
    oct_pas = sum(r["octets"] * r["couches"] for r in meilleurs.values() if r["couches"])
    ms_pas = sum(r["ms"] * r["couches"] for r in meilleurs.values() if r["couches"])
    pondere = oct_pas / (ms_pas * 1e-3) / 1e12 if ms_pas else float("nan")
    verdict = ("OUVRE (pondéré ≥ 1,0 To/s)" if pondere >= 1.0 else
               "FAUX (pondéré < 0,85 : noyau CUDA, décision séparée)" if pondere < 0.85 else
               "ENTRE 0,85 et 1,0 : à Sage")
    print(f"\nminimum sur les formes : {mini:.2f} To/s ; GEMM du pas (M = 12) : {oct_pas / 1e9:.1f} Go en "
          f"{ms_pas:.2f} ms → pondéré {pondere:.2f} To/s → {verdict}")
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "banc-gemm-dense-etroit-17-09.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0),
               "repet": REPET, "resultats": res, "meilleurs_m12": meilleurs, "min_To_s": mini,
               "pondere_To_s": pondere, "ms_gemm_pas": ms_pas, "verdict": verdict},
              open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
