"""GEMM W8A8 du préfill pour les linéaires INT8 denses (q/k/v/o de Coder :
`int8` affine par groupes de 128) — P0 (poste7-profil-verdict-18-09 § 1 (ii)) :
au-delà du seuil GEMV, `int8_matmul` déquantifiait la matrice ENTIÈRE en bf16
à chaque appel (`int8_dequant` 11,2 ms par préfill Coder 2 048) puis lançait
un GEMM bf16 cutlass (19,4 ms). Ici : l'activation est quantifiée en int8
PAR JETON (échelle absmax/127 par ligne, fp32), les poids restent uint8, et
la GEMM tourne sur les tensor cores int8 (`tl.dot` int8 → int32) sans
déquantification :

    y[m, n] = s_x[m] · Σ_g s_w[n, g] · ( Σ_{k∈g} a8[m, k]·(q[n, k] − 128)
                                        − (z[n, g] − 128) · Σ_{k∈g} a8[m, k] )

— le produit entier est exact (|Σ| ≤ 128·127·127), l'accumulation des groupes
en fp32 ; seule l'erreur est celle de l'A8 par jeton (porte PPL privé
≤ 1,020, prédiction poste7 ≤ 1,017). Régime ``ACVRAM_PREFILL_INT8 = bf16
(défaut : déquant + cutlass) | a8`` porté par `regime_ligne()`. Sans carte :
``TRITON_INTERPRET=1`` (le produit int8 → int32 y est exact aussi).
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

BM, BN = 128, 128
_WARPS, _STAGES = 8, 3
INTERPRETE = os.environ.get("TRITON_INTERPRET") == "1"


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available() or INTERPRETE)


if triton is not None:

    @triton.jit
    def _quant_a8_kernel(x_ptr, a_ptr, s_ptr, K, stride_x, stride_a, BK: tl.constexpr):
        m = tl.program_id(0)
        k = tl.arange(0, BK)
        masque = k < K
        x = tl.load(x_ptr + m * stride_x + k, mask=masque, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), 0)
        s = tl.maximum(amax, 1e-8) / 127.0
        v = x / s
        # arrondi au plus proche, identique sur carte et sous l'interpréteur
        r = tl.where(v >= 0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))
        r = tl.minimum(tl.maximum(r, -127.0), 127.0)
        tl.store(a_ptr + m * stride_a + k, r.to(tl.int8), mask=masque)
        tl.store(s_ptr + m, s)

    @triton.jit
    def _w8a8_kernel(a_ptr, q_ptr, s_ptr, z_ptr, sx_ptr, y_ptr, M, N, K, NG,
                     stride_am, stride_qn, stride_sn, stride_ym,
                     BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * BM_ + tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        kk = tl.arange(0, G)
        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        for g in range(0, NG):
            ks = g * G + kk
            masque_k = ks < K
            a = tl.load(a_ptr + rows[:, None] * stride_am + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0)          # int8 [BM, G]
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :],
                        mask=masque_n[:, None] & masque_k[None, :], other=128)       # uint8 [BN, G]
            qs = (q.to(tl.int16) - 128).to(tl.int8)                                  # symétrique
            prod = tl.dot(a, tl.trans(qs), out_dtype=tl.int32)                       # exact (int8 → int32)
            somme_a = tl.sum(a.to(tl.int32), 1)                                      # [BM]
            z = tl.load(z_ptr + cols * stride_sn + g, mask=masque_n, other=128).to(tl.int32) - 128
            s = tl.load(s_ptr + cols * stride_sn + g, mask=masque_n, other=0.0).to(tl.float32)
            acc += (prod - somme_a[:, None] * z[None, :]).to(tl.float32) * s[None, :]
        sx = tl.load(sx_ptr + rows, mask=masque_m, other=0.0)
        y = acc * sx[:, None]
        tl.store(y_ptr + rows[:, None] * stride_ym + cols[None, :], y.to(y_ptr.dtype.element_ty),
                 mask=masque_m[:, None] & masque_n[None, :])


def quantifier_a8(x: torch.Tensor):
    """``x`` [M, K] bf16/fp16/fp32 → (a8 int8 [M, K], s_x fp32 [M]) par jeton."""
    M, K = x.shape
    x = x.contiguous()
    a = torch.empty(M, K, dtype=torch.int8, device=x.device)
    s = torch.empty(M, dtype=torch.float32, device=x.device)
    BK = 1
    while BK < K:
        BK *= 2
    _quant_a8_kernel[(M,)](x, a, s, K, x.stride(0), a.stride(0), BK=max(BK, 16), num_warps=4)
    return a, s


def gemm_w8a8(x: torch.Tensor, t, sortie_fp32: bool = False) -> torch.Tensor:
    """``x`` [M, K], ``t`` INT8Tensor (qweight uint8 [N, K_pad], scales fp16
    [N, NG], zeros uint8 [N, NG], group_size G) → [M, N] dans le dtype de x."""
    M, K = x.shape
    N, k_pad = t.qweight.shape
    G = t.group_size
    assert k_pad % G == 0 and K <= k_pad, (K, k_pad, G)
    NG = k_pad // G
    a, sx = quantifier_a8(x)
    if K != k_pad:
        a = torch.nn.functional.pad(a, (0, k_pad - K))
    y = torch.empty(M, N, dtype=torch.float32 if sortie_fp32 else x.dtype, device=x.device)
    grille = (-(-M // BM), -(-N // BN))
    _w8a8_kernel[grille](a, t.qweight, t.scales, t.zeros, sx, y, M, N, k_pad, NG,
                         a.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0),
                         BM_=BM, BN_=BN, G=G, num_warps=_WARPS, num_stages=_STAGES)
    return y
