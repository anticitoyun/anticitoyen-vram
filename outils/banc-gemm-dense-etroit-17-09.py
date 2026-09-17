"""Porte micro-banc du GEMM dense étroit NVFP4 (poste7-hybrides-etape1-close-
gemm-dense-17-09 § 2) : octets de poids / temps, en rejeu de graphe, sur les
formes de Qwen3.8-27B à M = 12 (et 2, 32 en contrôle).

Scellé (poste7) : ≥ 1,3 To/s (73 % de 1,79) sur q/k/v/o et gate/up/down
OUVRE l'intégration ; < 0,9 To/s : Triton n'y arrive pas, noyau CUDA
(décision séparée). Témoins dans le même banc : la boucle GEMV actuelle
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
# (nom, N, K) — Qwen3.8-27B : H 5120, I 17408, 24 têtes q × 256, 4 kv ; GDN : qkv 10240
FORMES = [("q_proj", 6144, 5120), ("kv_proj", 2048, 5120), ("o_proj", 5120, 6144),
          ("gdn_qkv", 10240, 5120), ("gdn_out", 5120, 6144),
          ("gate_up", 34816, 5120), ("down", 5120, 17408)]
MS = [12] if "--rapide" in sys.argv else [2, 12, 32]
CONFIGS = [(64, 128, 4, 3), (64, 256, 4, 3), (128, 128, 4, 3), (32, 128, 4, 3),
           (64, 128, 8, 3), (64, 128, 4, 4), (64, 64, 4, 3), (128, 256, 8, 3)]
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
    for nom, N, K in FORMES:
        w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
        t = quantize_nvfp4(w)
        t.qweight, t.block_scale, t.global_scale = t.qweight.to(dev), t.block_scale.to(dev), t.global_scale.to(dev)
        octets = t.qweight.numel() + t.block_scale.numel()
        for M in MS:
            x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(dev)
            bras = {}
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
            # candidat : balayage des configurations
            for bn, bk, wp, st in CONFIGS:
                f = lambda: GD.gemm_dense_etroit(x, t, bn=bn, bk=bk, warps=wp, stages=st)
                try:
                    bras[f"triton_bn{bn}_bk{bk}_w{wp}_s{st}"] = (chrono(f), juger(f(), x, t), "")
                except Exception as exc:                          # noqa: BLE001
                    bras[f"triton_bn{bn}_bk{bk}_w{wp}_s{st}"] = (float("nan"), -1, type(exc).__name__)
            for b, (ms, hors, err) in bras.items():
                tos = octets / (ms * 1e-3) / 1e12 if ms == ms else float("nan")
                res.append({"forme": nom, "N": N, "K": K, "M": M, "bras": b, "ms": ms, "To_s": tos,
                            "part_crete": tos * 1e12 / BANDE_CRETE, "hors_2m7": hors, "erreur": err})
                print(f"{nom:8s} N={N:5d} K={K:5d} M={M:2d} {b:26s} {ms:8.3f} ms {tos:5.2f} To/s "
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
    verdict = ("OUVRE (≥ 1,3 To/s sur toutes les formes)" if mini >= 1.3 else
               "FAUX (< 0,9 : noyau CUDA, décision séparée)" if mini < 0.9 else
               "ENTRE 0,9 et 1,3 : à poste7")
    print(f"\nminimum sur les formes : {mini:.2f} To/s → {verdict}")
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "banc-gemm-dense-etroit-17-09.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0),
               "repet": REPET, "resultats": res, "meilleurs_m12": meilleurs, "min_To_s": mini, "verdict": verdict},
              open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
