#!/usr/bin/env python3
"""Poste E, noyau seul (5 min de carte) : attention paginée int8 CUDA
(`paged_attn_partial_kernel`) contre Triton (`kernels/attn_paginee.py`),
formes Coder-30B-A3B (48 couches, 32 têtes q / 4 têtes KV, D = 128), cache
int8 rempli au hasard, tables disjointes.

Scellé neuf (poste7-e-c-verdict-17-09) : distance fp64 de Triton ≤ 1,1 × celle
du CUDA sur 100 % des lignes (b, tête) — sinon vraie divergence ; le « hors
2⁻⁸ » entre deux sorties bf16 arrondies séparément vaut jusqu'à 1 ulp = 2⁻⁷
et ne juge rien. Vitesses (poste7), en REJEU DE GRAPHE (le régime servi ; l'eager, imprimé à
côté, mesure surtout le lanceur Python de Triton, 20-40 µs par appel) :
Triton ≤ 1,5 ms par pas (48 couches) à b = 12, ctx 2 048, ET ≤ 0,72 ms à b = 1 (ctx 300 : le cas servi le plus courant, aucune
régression) ; sortie = CUDA ± 2⁻⁸ (imprimé, lignes hors tolérance).

    outils/carte.sh python outils/banc-attn-paginee-17-09.py
"""
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.kernels import attn_paginee as ap, get_extension           # noqa: E402
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache          # noqa: E402

COUCHES, HQ, HKV, D = 48, 32, 4, 128
CELLULES = [(12, 2048, 1.5), (1, 300, 0.72), (1, 2048, None), (12, 300, None)]
REPET = 50


def montage(b, ctx, g):
    N = -(-ctx // 16)
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D,
                                   num_blocks=b * N, dtype="int8", device="cuda"))
    c.k.random_(-127, 128); c.v.random_(-127, 128)
    c.k_scale.uniform_(0.01, 0.05, generator=g); c.v_scale.uniform_(0.01, 0.05, generator=g)
    tables = torch.arange(b * N, device="cuda").view(b, N)
    lens = torch.full((b,), ctx, dtype=torch.long, device="cuda")
    q = torch.randn(b, HQ, D, device="cuda", generator=g).to(torch.bfloat16)
    return c, tables, lens, q


def reference64(c, tables, lens, q, scale):
    """Attention exacte (float64) lue du MÊME cache int8 : la référence de
    poste7 (poste7-e-c-verdict-17-09) — distance de chaque noyau à elle."""
    B, HQ, D = q.shape
    n_rep = HQ // HKV
    ref = torch.zeros(B, HQ, D, dtype=torch.float64, device=q.device)
    for b in range(B):
        n = int(lens[b])
        k, v = c.gather(tables[b], n, torch.float64)                 # [n, HKV, D]
        for h in range(HQ):
            kh, vh = k[:, h // n_rep], v[:, h // n_rep]
            p = torch.softmax((kh @ q[b, h].double()) * scale, 0)
            ref[b, h] = p @ vh
    return ref


def distances(y, ref):
    """Distance L2 par ligne (b, tête) à la référence, relative à la norme de la ligne."""
    return ((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)).reshape(-1)


def chrono(f):
    """Temps GPU d'un REJEU DE GRAPHE (le régime servi : graphes on, b=12) —
    pas d'un lancement eager : le lanceur Python de Triton coûte 20-40 µs par
    appel, et 4 linéaires × 48 couches en eager mesuraient le lanceur, pas le
    noyau (poste3 d65e49e : 6,59 ms « Triton » contre 4,86 CUDA). Un chiffre
    eager est imprimé aussi, pour le voir."""
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


def chrono_eager(f):
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
    ext = get_extension()
    g = torch.Generator(device="cuda").manual_seed(17)
    scale = 1 / math.sqrt(D)
    print(f"Coder-30B : {COUCHES} couches, HQ {HQ}, HKV {HKV}, D {D} ; ms par PAS = {COUCHES} × noyau")
    for b, ctx, seuil in CELLULES:
        c, tables, lens, q = montage(b, ctx, g)
        cuda = lambda: ext.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, lens, HKV, scale, 1, 0)
        tri = lambda: ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, lens, HKV, scale, 0)
        yc, yt = cuda(), tri()
        ecart = (yt.double() - yc.double()).abs().amax(-1)
        hors = int((ecart > yc.double().abs().amax(-1).clamp(min=1e-6) * 2 ** -8).sum())
        ref = reference64(c, tables, lens, q, scale)
        dc, dt = distances(yc, ref), distances(yt, ref)
        ratio = dt / dc.clamp(min=1e-12)
        ok11 = float((dt <= 1.1 * dc).float().mean())
        print(f"    distance à la référence fp64 (L2 relative par ligne) : cuda méd {dc.median():.2e} max {dc.max():.2e}"
              f" | triton méd {dt.median():.2e} max {dt.max():.2e} | triton ≤ 1,1×cuda sur {ok11:.1%} des lignes,"
              f" ratio max {ratio.max():.2f}, méd {ratio.median():.2f}")
        tc, tt = chrono(cuda) * COUCHES, chrono(tri) * COUCHES
        ec, et = chrono_eager(cuda) * COUCHES, chrono_eager(tri) * COUCHES
        C, chunk = ap._tranches(tables.shape[1], b, HKV, q.device)
        verdict = "" if seuil is None else ("TENU" if tt <= seuil else "HORS SCELLÉ")
        print(f"b={b:2d} ctx={ctx:5d}  graphe : cuda {tc:6.3f} ms/pas  triton {tt:6.3f} ms/pas (C={C}, chunk={chunk})"
              f"  ×{tc / tt:4.2f}  [eager cuda {ec:6.3f} triton {et:6.3f}]  hors 2^-8 : {hors}/{b * HQ}"
              + (f"  seuil {seuil} ms → {verdict}" if seuil else ""))


if __name__ == "__main__":
    main()
