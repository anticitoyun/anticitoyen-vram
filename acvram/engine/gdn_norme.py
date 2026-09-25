"""Pièce 156 F3 (opt-in `ACVRAM_GDN_NORME_FUSEE`) : norme gated de la sortie Gated DeltaNet en UN noyau.

Remplace `GatedDeltaNet._norm_gated` (pow, mean, +eps, rsqrt, trois produits, silu, cast bf16 : ≈ 8 lancements par
couche) par un programme par ligne de ``dv`` éléments, qui écrit le bf16 que l'appelant donnait à out_proj.

Même arithmétique que torch SAUF l'ordre de la somme des carrés (arbre de `tl.sum` contre la réduction de torch) :
± ulp fp32 sur la variance, donc KL et non « au bit » (scellé `revue/poste5-piece156d-scelle-24-09.md`).
* moyenne : somme × (1 / dv) — dv = 128, puissance de deux : la division de torch est la même opération ;
* rsqrt : `libdevice.rsqrt` (le `rsqrtf` de torch) ;
* silu : torch `z / (1 + expf(-z))` → `libdevice.exp` et `div_rn`, comme F2 (`gdn_conv.py`) ;
* ordre des produits : ((x · r) · w) · silu(z), puis bf16 au plus près.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norme_gated_kernel(x, z, w, y, sx, sz0, sz1, H, N, eps, BN: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BN)
    m = c < N
    xv = tl.load(x + r * sx + c, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(xv * xv, axis=0) / N
    xn = xv * libdevice.rsqrt(var + eps)
    xn = xn * tl.load(w + c, mask=m, other=0.0).to(tl.float32)
    zv = tl.load(z + (r // H) * sz0 + (r % H) * sz1 + c, mask=m, other=0.0).to(tl.float32)
    s = tl.math.div_rn(zv, 1.0 + libdevice.exp(-zv))
    tl.store(y + r * N + c, (xn * s).to(tl.bfloat16), mask=m)


def norme_gated(x: torch.Tensor, z: torch.Tensor, poids: torch.Tensor, eps: float) -> torch.Tensor:
    """``x`` [R, dv] ; ``z`` [R, dv] ou [b, H, dv] avec b·H = R (fp32 ou bf16, dernière dimension contiguë) ; ``poids``
    [dv]. Rend [R, dv] bf16. Pièce 182 : au décodage, z est la vue bf16 [b, nv, dv] de la pile qkv‖gate (pas de lot
    ≠ nv·dv), lue en place — le cast fp32 et la copie qu'il faisait (48 lancements par pas sur Qwen3.8) disparaissent ;
    la conversion bf16 → fp32 du chargement est exacte, donc au bit."""
    R, N = x.shape
    z3 = z if z.dim() == 3 else z.unsqueeze(1)
    assert x.stride(1) == 1 and z3.stride(2) == 1 and z3.shape[0] * z3.shape[1] == R and z3.shape[2] == N
    assert poids.is_contiguous()
    y = torch.empty(R, N, dtype=torch.bfloat16, device=x.device)
    if R:
        _norme_gated_kernel[(R,)](x, z3, poids, y, x.stride(0), z3.stride(0), z3.stride(1), z3.shape[1], N, eps,
                                  BN=triton.next_power_of_2(N), num_warps=1)
    return y
