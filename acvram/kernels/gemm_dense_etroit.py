"""GEMM W4A16 dense à petit M (2 ≤ M ≤ 32), bornée par la bande — le poste
révélé par la fenêtre fla (poste7-hybrides-etape1-close-gemm-dense-17-09 § 2) :
à b > 1, `nvfp4_gemv` relit les poids UNE FOIS PAR SÉQUENCE (Qwen3.8 b = 12 :
20 Go relus ligne par ligne à 0,23 To/s, 86,6 ms sur un pas de 93,9 ; plancher
de bande 11 ms).

Ici les M lignes d'activation forment une tuile [BM, BK] (M rembourré à 16 ou
32) qui reste en registres ; les poids NVFP4 sont balayés UNE fois par pas,
décodés en registres par les deux tables constantes de B1' (E2M1 → bf16,
E4M3 → bf16, produit code × échelle exact en bf16), et le produit passe par
``tl.dot`` sur la tuile — 0,65 TFLOP par pas sur Qwen3.8, la crête utile est
la bande, pas les tensor cores. Grille (tuiles N, tranches K) pour couvrir la
carte quand N est petit (q/k/v/o : N = 5 120 → 80 tuiles de 64) ; les
tranches K écrivent des partiels fp32 réduits par une somme torch
(déterministe, même ordre d'accumulation par tuile) ; échelle globale (par
ligne si le tenseur en porte) dans l'épilogue.

Porte : micro-banc `outils/banc-gemm-dense-etroit-17-09.py` ≥ 1,3 To/s sur les
formes de Qwen3.8 (faux < 0,9 : noyau CUDA, décision séparée). Régime
``ACVRAM_DENSE_NVFP4 = gemv (défaut, témoin) | triton``. Sans carte :
``TRITON_INTERPRET=1`` en fp16.
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

from .gemm_groupe import tables

M_MAX = 32
_BN = int(os.environ.get("ACVRAM_DENSE_ETROIT_BN", "64"))
_BK = int(os.environ.get("ACVRAM_DENSE_ETROIT_BK", "128"))
_WARPS = int(os.environ.get("ACVRAM_DENSE_ETROIT_WARPS", "4"))
_STAGES = int(os.environ.get("ACVRAM_DENSE_ETROIT_STAGES", "3"))
_PROGRAMMES_PAR_SM = 2          # tranches K visées : ≥ 2 programmes par SM


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _dense_etroit_kernel(x_ptr, qw_ptr, bs_ptr, gs_ptr, y_ptr, lut4_ptr, lut8_ptr,
                             M, N, K, k_par_tranche,
                             stride_xm, stride_qn, stride_bn, stride_gs, stride_yt, stride_ym,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pn = tl.program_id(0)
        pt = tl.program_id(1)
        rows = tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        masque_m = rows < M
        masque_n = cols < N
        k_debut = pt * k_par_tranche
        k_fin = tl.minimum(k_debut + k_par_tranche, K)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(k_debut, k_fin, BK):
            ks = k0 + tl.arange(0, BK)
            masque_k = ks < k_fin
            a = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            kb = k0 // 2 + tl.arange(0, BK // 2)
            masque_kb = kb < k_fin // 2
            oct = tl.load(qw_ptr + cols[:, None] * stride_qn + kb[None, :],
                          mask=masque_n[:, None] & masque_kb[None, :], other=0)
            lo = tl.load(lut4_ptr + (oct & 15))
            hi = tl.load(lut4_ptr + (oct >> 4))
            w = tl.reshape(tl.join(lo, hi), (BN, BK))
            ksc = k0 // 16 + tl.arange(0, BK // 16)
            masque_sc = ksc < k_fin // 16
            sc = tl.load(bs_ptr + cols[:, None] * stride_bn + ksc[None, :],
                         mask=masque_n[:, None] & masque_sc[None, :], other=0)
            scb = tl.load(lut8_ptr + sc)
            scf = tl.reshape(tl.broadcast_to(tl.expand_dims(scb, 2), (BN, BK // 16, 16)), (BN, BK))
            b = (w * scf).to(a.dtype)
            acc = tl.dot(a, tl.trans(b), acc)
        gs = tl.load(gs_ptr + cols * stride_gs, mask=masque_n, other=0.0)
        acc = acc * gs[None, :]
        tl.store(y_ptr + pt * stride_yt + rows[:, None] * stride_ym + cols[None, :],
                 acc, mask=masque_m[:, None] & masque_n[None, :])


def _sms(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4


def _tranches(n: int, k: int, device, bn: int, bk: int) -> tuple[int, int]:
    """(tranches K, jetons K par tranche) : assez de programmes pour couvrir
    la carte, tranche multiple de BK (les octets de codes et d'échelles d'une
    tranche restent alignés sur 2 et 16)."""
    tuiles_n = -(-n // bn)
    voulu = max(1, -(-_PROGRAMMES_PAR_SM * _sms(device) // tuiles_n))
    pas_max = -(-k // bk)
    t = max(1, min(voulu, pas_max))
    par_tranche = -(-pas_max // t) * bk
    return -(-k // par_tranche), par_tranche


def gemm_dense_etroit(x: torch.Tensor, t, bn: int = 0, bk: int = 0, warps: int = 0,
                      stages: int = 0) -> torch.Tensor:
    """``x`` [M ≤ 32, K] bf16 (fp16 sous l'interpréteur), ``t`` NVFP4Tensor
    (échelle globale scalaire ou par ligne) → [M, N] dans le dtype de x."""
    bn, bk = bn or _BN, bk or _BK
    warps, stages = warps or _WARPS, stages or _STAGES
    M, K = x.shape
    N, k_oct = t.qweight.shape
    k_pad = t.padded_in
    assert 1 <= M <= M_MAX and K <= k_pad and k_pad % 16 == 0 and bk % 16 == 0, (M, K, k_pad, bk)
    if K != k_pad:
        x = torch.nn.functional.pad(x, (0, k_pad - K))
    x = x.contiguous()
    BM = 16 if M <= 16 else 32
    lut4, lut8 = tables(x.device, x.dtype)
    gsr = getattr(t, "global_scale_rows", None)
    gs = (gsr.to(torch.float32).reshape(-1).contiguous() if gsr is not None
          else t.global_scale.to(torch.float32).reshape(1))
    tuiles_n = -(-N // bn)
    tranches, par_tranche = _tranches(N, k_pad, x.device, bn, bk)
    y = torch.empty(tranches, M, N, dtype=torch.float32, device=x.device)
    _dense_etroit_kernel[(tuiles_n, tranches)](
        x, t.qweight, t.block_scale.view(torch.uint8), gs, y, lut4, lut8,
        M, N, k_pad, par_tranche,
        x.stride(0), t.qweight.stride(0), t.block_scale.stride(0), 1 if gsr is not None else 0,
        y.stride(0), y.stride(1),
        BM=BM, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    out = y.sum(0) if tranches > 1 else y[0]
    return out.to(x.dtype)
