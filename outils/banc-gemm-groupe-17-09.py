#!/usr/bin/env python3
"""Porte de B1' (poste7-b1-verdict-17-09) : le noyau seul, 5 min de carte.

Formes du prefill Coder-30B-A3B (E = 128, 2 048 jetons × top-8 = 16 384
lignes, comptes tirés au sort autour de 128 par expert) : gate/up
K = 2 048 → M = 768, down K = 768 → M = 2 048. Trois bras sur les mêmes
entrées : B0 (`gemm_groupe`, pile bf16 déjà déquantifiée), B1' (`gemm_groupe_nvfp4`,
NVFP4 dans la tuile), et `torch._grouped_mm` (témoin). TFLOPS = 2·G·K·M / t.

Porte : B1' ≥ 85 TFLOPS sur gate/up, sinon le chantier prefill est clos à B0
(poste7, pas de 3e essai). Le juge d'équivalence tourne d'abord sur ces entrées.

    outils/carte.sh python outils/banc-gemm-groupe-17-09.py
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.engine.model import MoEBlock                     # noqa: E402
from acvram.kernels import gemm_groupe as gg                 # noqa: E402
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4  # noqa: E402

E, T, TOPK = 128, 2048, 8
FORMES = {"gate/up": (2048, 768), "down": (768, 2048)}
REPET = 20


def pile(M, K, g):
    ts = [quantize_nvfp4((torch.randn(M, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16))
          for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous()
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous()
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).float().contiguous()
    w = torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts]).contiguous()
    return qw, bs, gs, w


def chrono(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPET):
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); f(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a))
    return statistics.median(ts)


def main():
    g = torch.Generator(device="cuda").manual_seed(17)
    cnt = torch.multinomial(torch.rand(E, device="cuda", generator=g) + 0.2, T * TOPK, replacement=True).bincount(minlength=E)
    G = int(cnt.sum())
    print(f"E={E} G={G} comptes min/méd/max {int(cnt.min())}/{int(cnt.float().median())}/{int(cnt.max())} "
          f"tuile {gg.BT}×{gg._BN}×{gg._BK} warps {gg._WARPS} étages B0 {gg._STAGES} / B1' {gg._STAGES_NVFP4}")
    tiles = MoEBlock._tuiles(cnt, gg.BT, t_max=-(-G // gg.BT) + E)
    offs = torch.cumsum(cnt, 0).to(torch.int32)
    for nom, (K, M) in FORMES.items():
        qw, bs, gs, w = pile(M, K, g)
        xs = torch.randn(G, K, device="cuda", generator=g).to(torch.bfloat16)
        y0 = gg.gemm_groupe(xs, w, tiles)
        y1 = gg.gemm_groupe_nvfp4(xs, qw, bs, gs, tiles)
        ecart = float((y1.float() - y0.float()).abs().max() / y0.float().abs().max())
        flop = 2 * G * K * M
        res = {}
        res["B0 bf16"] = chrono(lambda: gg.gemm_groupe(xs, w, tiles))
        res["B1' nvfp4"] = chrono(lambda: gg.gemm_groupe_nvfp4(xs, qw, bs, gs, tiles))
        if hasattr(torch, "_grouped_mm"):
            res["grouped_mm"] = chrono(lambda: torch._grouped_mm(xs, w.transpose(1, 2), offs=offs))
        print(f"{nom:8s} K={K} M={M}  écart max B1'/B0 {ecart:.2e}")
        for k, ms in res.items():
            print(f"    {k:12s} {ms:8.3f} ms  {flop / ms / 1e9:7.1f} TFLOPS")
    porte = 85.0
    print(f"porte B1' : ≥ {porte} TFLOPS sur gate/up ; sinon clos à B0 (poste7-b1-verdict-17-09)")


if __name__ == "__main__":
    main()
