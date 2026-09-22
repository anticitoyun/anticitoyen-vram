#!/usr/bin/env python3
"""Sonde (carte, 10 s) : où le noyau rope_kv perd le demi-entier exact.

Pour les trois cas de Laure (x = amax/2 en bf16 → x·(1/sc) = 63,5 exactement
en fp32 IEEE, numpy et torch CPU le confirment), un noyau Triton minimal
imprime chaque intermédiaire calculé SUR LA CARTE, en bits : sc = amax/127
(fp32 IEEE via fp64), 1/sc par trois voies (`/`, `tl.fdiv(ieee_rounding=True)`,
fp64→fp32), les trois produits x·inv, et les étapes de `_rint` (floor, r,
parité, résultat) sur le produit fp64. Ce qui diffère de la valeur attendue
est la cause.

    outils/carte.sh python outils/sonde-rint-17-09.py
"""
import struct
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton                                              # noqa: E402
import triton.language as tl                               # noqa: E402
from acvram.kernels.rope_kv import _rint                   # noqa: E402

CAS = [(1.2421875, 2.484375), (1.4921875, 2.984375), (-0.9453125, 1.890625)]
NOMS = ["sc", "inv_div", "inv_fdiv_ieee", "inv_fp64", "p_div", "p_fdiv", "p_fp64",
        "floor", "r", "impair", "rint_maison", "rint_libdevice"]


@triton.jit
def _sonde(x_ptr, a_ptr, out_ptr, N: tl.constexpr, NC: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(x_ptr + i)
    a = tl.load(a_ptr + i)
    sc = tl.maximum((a.to(tl.float64) / 127.0).to(tl.float32), 1e-8)
    inv_div = 1.0 / sc
    inv_fdiv = tl.fdiv(1.0, sc, ieee_rounding=True)
    inv_64 = (1.0 / sc.to(tl.float64)).to(tl.float32)
    p_div = x * inv_div
    p_fdiv = x * inv_fdiv
    p_64 = x * inv_64
    f = tl.floor(p_64)
    r = p_64 - f
    impair = ((f - 2.0 * tl.floor(f * 0.5)) == 1.0).to(tl.float32)
    rm = _rint(p_64)
    rl = tl.extra.cuda.libdevice.rint(p_64)
    vals = [sc, inv_div, inv_fdiv, inv_64, p_div, p_fdiv, p_64, f, r, impair, rm, rl]
    for c in tl.static_range(NC):
        tl.store(out_ptr + i * NC + c, vals[c])


def bits(v):
    return "0x%08x" % struct.unpack("<I", struct.pack("<f", v))[0]


def main():
    x = torch.tensor([c[0] for c in CAS], dtype=torch.float32, device="cuda")
    a = torch.tensor([c[1] for c in CAS], dtype=torch.float32, device="cuda")
    out = torch.zeros(len(CAS), len(NOMS), dtype=torch.float32, device="cuda")
    _sonde[(1,)](x, a, out, N=4 if len(CAS) <= 4 else 8, NC=len(NOMS))
    out = out.cpu()
    for k, (xv, av) in enumerate(CAS):
        sc = torch.tensor(av) / 127
        inv = 1.0 / sc
        p = torch.tensor(xv) * inv
        print(f"cas {k}: x={xv} amax={av} | attendu (torch cpu) sc={sc.item()!r} {bits(sc.item())} "
              f"inv={inv.item()!r} {bits(inv.item())} produit={p.item()!r} {bits(p.item())} rint={p.round().item()}")
        for j, nom in enumerate(NOMS):
            v = out[k, j].item()
            print(f"    {nom:15s} {v!r:>22} {bits(v)}")


if __name__ == "__main__":
    main()
