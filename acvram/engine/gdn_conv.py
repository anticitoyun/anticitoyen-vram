"""Pièce 156 F2 (opt-in `ACVRAM_GDN_CONV_FUSEE`) : convolution causale du décodage Gated DeltaNet en UN noyau.

Remplace, pour ``b`` jetons (un par séquence), le cast fp32 de qkv, le `cat` avec l'état, la recopie de l'état,
`conv1d` depthwise, `silu`, la découpe q/k/v et les deux `repeat_interleave` (`gdn.py`, `_lot_projete`).

Visé AU BIT contre torch, donc écrit en Triton : l'extension CUDA est compilée en `--use_fast_math`
(`kernels/__init__.py`), où `expf` et la division ne sont plus ceux de torch.
* conv : torch (`conv_depthwise2d_forward_kernel_generic`, fp32) part de 0 et accumule prise par prise, dans l'ordre,
  `value += w * x` contracté en FMA par nvcc → ici `tl.fma` explicite, même ordre ;
* silu : torch `x / (1 + expf(-x))` → `libdevice.exp` (le `expf` précis) et `div_rn` (division IEEE) ;
* q et k SANS répétition des têtes : fla lit la tête `i_hv // (HV // H)` (`fused_recurrent.py`), la disposition exacte
  de `repeat_interleave` — mêmes valeurs lues.
Le test au bit (`tests/test_gdn_fusions_au_bit.py`) tranche ; s'il tombe, F2 n'est pas livrée.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _conv_decode_kernel(qkv, sx, etat, w, q, k, v, C, KD, VD, KER: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    x_neuf = tl.load(qkv + r * sx + c, mask=m, other=0.0).to(tl.float32)
    base = etat + (r * C + c) * (KER - 1)
    acc = tl.zeros([BC], dtype=tl.float32)
    for j in tl.static_range(KER - 1):
        s = tl.load(base + j, mask=m, other=0.0)
        acc = tl.fma(tl.load(w + c * KER + j, mask=m, other=0.0), s, acc)
    acc = tl.fma(tl.load(w + c * KER + (KER - 1), mask=m, other=0.0), x_neuf, acc)
    # état décalé d'une colonne, en place : [s1, …, s_{K-2}, x]
    for j in tl.static_range(KER - 2):
        tl.store(base + j, tl.load(base + j + 1, mask=m, other=0.0), mask=m)
    tl.store(base + (KER - 2), x_neuf, mask=m)
    y = tl.math.div_rn(acc, 1.0 + libdevice.exp(-acc))
    tl.store(q + r * KD + c, y, mask=m & (c < KD))
    tl.store(k + r * KD + (c - KD), y, mask=m & (c >= KD) & (c < 2 * KD))
    tl.store(v + r * VD + (c - 2 * KD), y, mask=m & (c >= 2 * KD))


def conv_decode(qkv: torch.Tensor, etat: torch.Tensor, poids: torch.Tensor, key_dim: int, value_dim: int):
    """``qkv`` [b, C] (bf16 ou fp32, lignes éventuellement espacées), ``etat`` [b, C, K−1] fp32 contigu (mis à jour EN
    PLACE), ``poids`` [C, K] fp32. Rend q, k [b, key_dim], v [b, value_dim] fp32 contigus."""
    b, C = qkv.shape
    assert qkv.stride(1) == 1 and etat.is_contiguous() and poids.is_contiguous() and etat.dtype == torch.float32
    q = torch.empty(b, key_dim, dtype=torch.float32, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty(b, value_dim, dtype=torch.float32, device=qkv.device)
    BC = 256
    _conv_decode_kernel[(b, triton.cdiv(C, BC))](qkv, qkv.stride(0), etat, poids, q, k, v, C, key_dim, value_dim,
                                                  KER=poids.shape[1], BC=BC)
    return q, k, v
