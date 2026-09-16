"""GEMM étroit W8A16 en Triton — poste C (poste7-profil-verdict-17-09 § 2) :
les linéaires INT8 denses du décodage à b ≤ 16 (`narrow_gemm_kernel<32>`
2,08 ms pour 0,9 Go à 24 % de la bande passante, `int8_gemv` de la tête
0,88 ms pour 0,31 Go — poste3 6a4fd57).

Poids uint8 affine par groupes (`INT8Tensor` : q [N, K], échelle fp16 et
zéro uint8 [N, K/G]) : par groupe g, ``y += (x_g · q_gᵀ − Σx_g · z_g) · s_g``
— le produit se fait sur les entiers (uint8 ≤ 255 exact en bf16) par
``tl.dot`` en fp32, le zéro ne coûte qu'une somme de ligne, l'échelle un
produit par colonne. Grille (tuile N, tranche K) pour couvrir la carte à
b = 1 comme à b = 12 ; tranches réduites par une somme torch (déterministe).
Sortie bf16, ou fp32 pour la tête (logits).

Scellé (poste7) : dense b = 12 ≤ 1,0 ms par pas (−1,8 ms), PPL inchangée ;
sortie = chemin actuel ± 2⁻⁸ (juge : tests/test_gemm_etroit.py). Sans
carte : ``TRITON_INTERPRET=1`` en fp16.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None

BM = 16                    # lignes de la tuile : b ≤ 16 rembourré
BN = 64
_WARPS, _STAGES = 4, 3


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _etroit_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, M, N, K, groupes_par_tranche,
                       stride_xm, stride_qn, stride_sn, stride_ys, stride_ym,
                       BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        kk = tl.arange(0, G)
        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        g0 = ps * groupes_par_tranche
        for g in range(g0, g0 + groupes_par_tranche):
            ks = g * G + kk
            masque_k = ks < K
            x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :],
                        mask=masque_n[:, None] & masque_k[None, :], other=0)
            s = tl.load(s_ptr + cols * stride_sn + g, mask=masque_n, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + cols * stride_sn + g, mask=masque_n, other=0).to(tl.float32)
            prod = tl.dot(x, tl.trans(q.to(x.dtype)))                     # [BM, BN] fp32
            sx = tl.sum(x.to(tl.float32), 1)                              # Σ_k x[m, k] du groupe
            acc += (prod - sx[:, None] * z[None, :]) * s[None, :]
        tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                 acc, mask=masque_m[:, None] & masque_n[None, :])


def _programmes(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4


def gemm_etroit(x: torch.Tensor, t, sortie_fp32: bool = False) -> torch.Tensor:
    """``x`` [M ≤ 16, K] bf16 (fp16 sous l'interpréteur), ``t`` INT8Tensor
    → [M, N] dans le dtype de x, ou fp32 (tête)."""
    M, K = x.shape
    N, k_pad = t.qweight.shape
    G = t.group_size
    assert M <= BM and K <= k_pad and k_pad % G == 0, (M, K, k_pad, G)
    ng = k_pad // G
    tuiles_n = -(-N // BN)
    voulu = -(-2 * _programmes(x.device) // tuiles_n)
    tranches = max(1, min(ng, voulu))
    gpt = -(-ng // tranches)
    tranches = -(-ng // gpt)
    y = torch.zeros(tranches, M, N, dtype=torch.float32, device=x.device)
    _etroit_kernel[(tuiles_n, tranches)](
        x, t.qweight, t.scales, t.zeros, y, M, N, K, gpt,
        x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1),
        BM_=BM, BN_=BN, G=G, num_warps=_WARPS, num_stages=_STAGES)
    out = y.sum(0) if tranches > 1 else y[0]
    return out if sortie_fp32 else out.to(x.dtype)
