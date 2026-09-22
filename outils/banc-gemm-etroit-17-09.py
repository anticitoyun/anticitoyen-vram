#!/usr/bin/env python3
"""Poste C, noyau seul (5 min de carte) : linéaires INT8 denses à b = 12 et
b = 1 — `narrow_gemm` / `int8_gemv` (CUDA, chemin actuel) contre
`gemm_etroit` (Triton). Formes Coder-30B-A3B dense par couche (q 2048→4096,
k/v 2048→512 ×2, o 4096→2048, groupes de 128) × 48 couches, et la tête
2048→151 936 (fp32). Scellé : dense b = 12 ≤ 1,0 ms par pas (Sage), en REJEU DE GRAPHE (le régime
servi) ; l'eager est imprimé à côté et mesure surtout le lanceur Python de
Triton ; sortie = CUDA ± 2⁻⁸ (imprimé).

    outils/carte.sh python outils/banc-gemm-etroit-17-09.py
"""
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import acvram.kernels as kernels                                            # noqa: E402
from acvram.kernels import gemm_etroit as ge, get_extension, int8_matmul     # noqa: E402

# Le bras « cuda » FORCE le noyau CUDA : `int8_matmul` suit ACVRAM_NARROW_KERNEL,
# et sous le défaut `mixte` les deux bras rendaient le même noyau Triton
# (2,028 / 2,031 ms — un témoin qui ne témoignait pas, Laure a9e5f59).
kernels._NARROW_KERNEL = "cuda"
assert kernels.narrow_choix(12) == "cuda" and kernels.narrow_choix(1) == "cuda"
from acvram.quant.formats import _quantize_int8                             # noqa: E402

COUCHES = 48
DENSE = [("q_proj", 4096, 2048), ("k_proj", 512, 2048), ("v_proj", 512, 2048), ("o_proj", 2048, 4096)]
TETE = ("lm_head", 151936, 2048)
REPET = 30


def tenseur(n, k, g):
    t = _quantize_int8(torch.randn(n, k, generator=g) * 0.05, group_size=128)
    t.qweight, t.scales, t.zeros = t.qweight.cuda(), t.scales.cuda(), t.zeros.cuda()
    return t


def chrono(f):
    """Temps GPU d'un REJEU DE GRAPHE (le régime servi : graphes on, b=12) —
    pas d'un lancement eager : le lanceur Python de Triton coûte 20-40 µs par
    appel, et 4 linéaires × 48 couches en eager mesuraient le lanceur, pas le
    noyau (Laure d65e49e : 6,59 ms « Triton » contre 4,86 CUDA). Un chiffre
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
    assert get_extension() is not None
    g = torch.Generator().manual_seed(17)
    for b in tuple(int(x) for x in os.environ.get("BANC_B", "12,1").split(",")):   # BANC_B=1,2,4,8,12 : bascule cuda/Triton (Sage, régime mixte)
        total_c = total_t = 0.0
        hors_tot = 0
        for nom, n, k in DENSE:
            t = tenseur(n, k, g)
            x = torch.randn(b, k, generator=g).cuda().to(torch.bfloat16)
            os.environ["ACVRAM_NARROW_GEMM"] = "1"
            cuda = lambda: int8_matmul(x, t)
            tri = lambda: ge.gemm_etroit(x, t)
            yc, yt = cuda(), tri()
            borne = x.float().abs() @ (t.scales.float().repeat_interleave(128, 1) * 255).abs().T
            hors_tot += int(((yt.float() - yc.float()).abs() > 2 ** -8 * borne).sum())
            tc, tt = chrono(cuda), chrono(tri)
            total_c += tc; total_t += tt
            print(f"b={b:2d} {nom:7s} {n:6d}x{k:5d}  graphe : cuda {tc:7.4f} ms  triton {tt:7.4f} ms  ×{tc / tt:4.2f}"
                  f"   eager : cuda {chrono_eager(cuda):7.4f}  triton {chrono_eager(tri):7.4f}")
        t = tenseur(*TETE[1:], g)
        x = torch.randn(b, TETE[2], generator=g).cuda().to(torch.bfloat16)
        cuda = lambda: int8_matmul(x, t, sortie_fp32=True)
        tri = lambda: ge.gemm_etroit(x, t, sortie_fp32=True)
        tc, tt = chrono(cuda), chrono(tri)
        print(f"b={b:2d} {TETE[0]:7s} {TETE[1]:6d}x{TETE[2]:5d}  cuda {tc:7.4f} ms  triton {tt:7.4f} ms  ×{tc / tt:4.2f}")
        pas_c, pas_t = total_c * COUCHES + tc, total_t * COUCHES + tt
        print(f"b={b:2d} PAS dense ({COUCHES} couches + tête) : cuda {pas_c:6.3f} ms  triton {pas_t:6.3f} ms"
              f"  hors 2^-8 : {hors_tot}" + (f"  seuil 1,0 ms → {'TENU' if pas_t <= 1.0 else 'HORS SCELLÉ'}" if b == 12 else ""))


if __name__ == "__main__":
    main()
