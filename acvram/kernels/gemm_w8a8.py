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


def quantifier_a8_torch(x: torch.Tensor):
    """Le MÊME arrondi que `_quant_a8_kernel`, en torch pur (à sec, fausse quant,
    repli cublas sans Triton) : s = max(amax, 1e-8)/127 par ligne en fp32,
    arrondi au plus proche À DEMI ÉLOIGNÉ DE ZÉRO (floor(v + 0,5) selon le
    signe — pas `torch.round`, qui arrondit au pair), borne ± 127.
    `tests/test_prefill_a8_porte.py` le tient au bit contre le noyau."""
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    s = torch.clamp_min(xf.abs().amax(dim=1), 1e-8) / 127.0
    v = xf / s[:, None]
    r = torch.where(v >= 0, torch.floor(v + 0.5), -torch.floor(-v + 0.5)).clamp(-127.0, 127.0)
    return r.to(torch.int8), s


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


# --- C15-prefill : épilogue du chemin cublas (i8c) en un lancement ------------
# `gemm_i8c_cublas` (kernels/__init__.py) rendait y = bf16(f32(acc) · s_x[m] · s_w[n])
# par QUATRE noyaux torch sur [M, N] entiers (nsys P2 19/09 : direct_copy int32→f32
# ×4/couche, MulFunctor<float> ×8/couche = 3,8 ms, bfloat16_copy ×4/couche = 0,7 ms —
# 6,5 ms de noyaux et 16 lancements par couche pour 88 Mo utiles). Ici la même
# chaîne d'arrondis dans l'ordre : cvt int32→fp32 (RN), × s_x (fp32, RN), × s_w
# (fp32, RN), → bf16 (RNE) — chaque opération est celle de torch, dans le même
# ordre, donc les octets sont les mêmes ; `tests/test_prefill_compact.py` le
# tient au bit et casse si l'ordre des produits change (témoin).
EBM, EBN = 64, 128


if triton is not None:

    @triton.jit
    def _epilogue_i8c_kernel(acc_ptr, sx_ptr, sw_ptr, y_ptr, M, N, stride_am, stride_ym,
                             BM_: tl.constexpr, BN_: tl.constexpr, BF16: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rows = pm * BM_ + tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque = (rows < M)[:, None] & (cols < N)[None, :]
        acc = tl.load(acc_ptr + rows[:, None] * stride_am + cols[None, :], mask=masque, other=0)
        sx = tl.load(sx_ptr + rows, mask=rows < M, other=0.0)
        sw = tl.load(sw_ptr + cols, mask=cols < N, other=0.0)
        y = acc.to(tl.float32) * sx[:, None]              # f32(acc) · s_x : premier arrondi
        y = y * sw[None, :]                               # · s_w : second arrondi
        ptr = y_ptr + rows[:, None] * stride_ym + cols[None, :]
        if BF16:
            # bf16 au plus proche, pair en cas d'égalité — la formule de
            # c10::BFloat16 (bits + 0x7FFF + bit 16, puis >> 16), écrite en
            # entiers : la même sur carte et sous l'interpréteur, qui, lui,
            # TRONQUE `.to(tl.bfloat16)` (vérifié 20/09 : 659,4 → 656 au lieu
            # de 660) ; un NaN n'arrive pas ici (produit d'un entier fini et
            # d'échelles finies)
            b = y.to(tl.int32, bitcast=True)
            b = b + 0x7FFF + ((b >> 16) & 1)
            tl.store(ptr, (b >> 16).to(tl.int16), mask=masque)
        else:
            tl.store(ptr, y, mask=masque)


def epilogue_i8c(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``acc`` int32 [M, N] (torch._int_mm), ``sx`` fp32 [M], ``sw`` fp32 [N] →
    ``dtype(f32(acc) · sx[:, None] · sw[None, :])`` en un lancement — au bit
    avec `epilogue_i8c_torch` ; dtype bf16 ou fp32."""
    M, N = acc.shape
    assert dtype in (torch.bfloat16, torch.float32), dtype
    assert sx.dtype == torch.float32 and sw.dtype == torch.float32 and sx.numel() == M and sw.numel() == N
    y = torch.empty(M, N, dtype=dtype, device=acc.device)
    bf16 = dtype == torch.bfloat16
    grille = (-(-M // EBM), -(-N // EBN))
    _epilogue_i8c_kernel[grille](acc, sx.contiguous(), sw.contiguous(), y.view(torch.int16) if bf16 else y,
                                 M, N, acc.stride(0), y.stride(0), BM_=EBM, BN_=EBN, BF16=bf16, num_warps=4)
    return y


def epilogue_i8c_torch(acc: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Le témoin : la chaîne torch d'avant C15-prefill, quatre noyaux (kernels/
    __init__.py, `gemm_i8c_cublas` sous PREFILL_COMPACT=0)."""
    y = acc.to(torch.float32) * sx[:, None] * sw[None, :]
    return y if dtype == torch.float32 else y.to(dtype)
