"""Pièce 175 (poste6, 25/09) : GEMM étroite bf16 pour M ≤ 16 lignes, ``y = x @ W.T`` avec accumulation fp32 sur
tensor cores (tl.dot), un programme par bloc de BN colonnes, K parcouru en entier par chaque programme — donc
DÉTERMINISTE (aucun atomique), mais PAS au bit de cuBLAS (ordre de réduction différent).

Pourquoi : au décodage b=8 de l'alias mixte, les portes α et β des 48 couches GDN (poids bf16 [48, 5120], `PlainTensor`)
passaient par `F.linear` ; cuBLAS y lance 4 blocs × 32 fils, 31 µs pour 0,5 Mo (trouvaille nsys d'poste1, 173) : 96
appels = 2,99 ms sur 19,9 par pas (15 %). Ici : W [96, 5120] lu une fois par bloc, ~3-6 µs."""
from __future__ import annotations

import torch

# Pièce 267c (CI GitHub, runner sans triton) : `gemv_bf16_etroit()` ci-dessous retombe
# DÉJÀ sur `F.linear` quand `not x.is_cuda` (ligne « if M > 16 or not x.is_cuda... ») —
# triton n'est donc jamais réellement nécessaire sur CPU. Seul l'import inconditionnel
# (et le `@triton.jit` sur le noyau) empêchait ce fichier de se charger sans triton
# installé, avant même que la garde CPU n'ait sa chance. Même motif que marlin_port.py.
try:
    import triton
    import triton.language as tl
except Exception:                                          # noqa: BLE001
    triton = None
    tl = None

if triton is not None:

    @triton.jit
    def _gemv_bf16_kernel(x, w, y, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          ARRONDI_BF16: tl.constexpr):
        n0 = tl.program_id(0) * BN
        rm = tl.arange(0, BM)
        rn = n0 + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            xk = tl.load(x + rm[:, None] * sxm + (k0 + rk)[None, :], mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
            wk = tl.load(w + rn[:, None] * swn + (k0 + rk)[None, :], mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < K), other=0.0)
            acc = tl.dot(xk, tl.trans(wk), acc)                       # [BM, BK] · [BK, BN], fp32
        if ARRONDI_BF16:                                              # sortie fp32 AU BIT de « bf16 puis .to(float32) » (175, casts)
            acc = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(y + rm[:, None] * sym + rn[None, :], acc.to(y.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def gemv_bf16_etroit(x: torch.Tensor, w: torch.Tensor, fp32: bool = False) -> torch.Tensor:
    """``x`` [M, K] bf16 (M ≤ 16), ``w`` [N, K] bf16 contigu → [M, N] bf16 ; ``fp32`` : sortie fp32 égale au bit à
    ``(...).to(torch.float32)`` (arrondi bf16 dans le registre), sans le lancement du cast. Au-delà de 16 lignes : F.linear."""
    M, K = x.shape
    N = w.shape[0]
    if M > 16 or not x.is_cuda or K % 16 != 0:
        y = torch.nn.functional.linear(x, w)
        return y.to(torch.float32) if fp32 else y
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.float32 if fp32 else x.dtype, device=x.device)
    BN = 32 if N % 32 == 0 else 16
    _gemv_bf16_kernel[(triton.cdiv(N, BN),)](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0),
                                             BM=16, BN=BN, BK=128, ARRONDI_BF16=fp32, num_warps=4)
    return y
