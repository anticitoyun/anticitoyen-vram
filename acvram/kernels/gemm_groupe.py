"""GEMM groupée bf16 persistante en Triton — marche B0 du prefill MoE
(poste7-prefill-b-plan-17-09 § 2) : UN lancement pour tous les experts.

Le coût du prefill MoE n'est ni la synchronisation ni la tuile (A réfuté,
poste3 0edc3b9) mais 3 × 128 GEMM séquentielles par couche, chacune trop
petite pour occuper la carte. Ici la grille est persistante (un programme
par SM, chacun consomme des tuiles (expert, bloc de lignes, bloc de
colonnes) jusqu'à épuisement), les lignes sont lues par index dans le
noyau — aucune copie ``w[experts]``, aucun ``offs`` relu sur l'hôte : la
grille de tuiles ``(te, t0, tn)`` est celle de ``MoEBlock._tuiles`` à
taille fixe, calculée sur la carte.

Marche B1 (à venir) : même noyau, opérande B chargé en E2M1 + E4M3 et
déquantifié en registres avant ``tl.dot``.

Juge : tests/test_gemm_grouped_w4a16.py (référence float64, 2⁻⁷ × Σ|x·w|,
un offset décalé d'une ligne casse). Sans carte : ``TRITON_INTERPRET=1`` — en fp16 : l interpréteur (numpy) rend
n importe quoi en bf16, le noyau est écrit indépendant du type.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

try:                                                     # Triton 3.8 dans le venv
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _gemm_groupe_kernel(xs_ptr, w_ptr, y_ptr, te_ptr, t0_ptr, tn_ptr,
                            n_tuiles, M, K,
                            stride_xg, stride_we, stride_wm, stride_yg,
                            BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        n_blocs_n = tl.cdiv(M, BN)
        total = n_tuiles * n_blocs_n
        # ordre (tuile, bloc N) : les programmes voisins lisent le même bloc
        # de lignes et des colonnes différentes — les lignes restent en L2
        for t in range(pid, total, nprog):
            tuile = t // n_blocs_n
            nb = t % n_blocs_n
            e = tl.load(te_ptr + tuile)
            t0 = tl.load(t0_ptr + tuile)
            n = tl.load(tn_ptr + tuile)
            lignes = tl.arange(0, BT)
            masque_l = lignes < n                         # n = 0 : tuile fantôme, rien n'est lu ni écrit
            rows = t0 + lignes
            cols = nb * BN + tl.arange(0, BN)
            masque_c = cols < M
            acc = tl.zeros((BT, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                masque_k = ks < K
                a = tl.load(xs_ptr + rows[:, None] * stride_xg + ks[None, :],
                            mask=masque_l[:, None] & masque_k[None, :], other=0.0)
                b = tl.load(w_ptr + e * stride_we + cols[:, None] * stride_wm + ks[None, :],
                            mask=masque_c[:, None] & masque_k[None, :], other=0.0)
                acc = tl.dot(a, tl.trans(b), acc)
            tl.store(y_ptr + rows[:, None] * stride_yg + cols[None, :],
                     acc.to(y_ptr.dtype.element_ty), mask=masque_l[:, None] & masque_c[None, :])


BT = 128                # lignes par tuile : celui de `_tuiles(cnt, BT)`
# Tuile 128 × 128 × 64, 8 warps, 2 étages : 2 × (128 + 128) × 64 × 2 o = 64 Kio
# de mémoire partagée — sous les ~99 Kio par bloc de sm_120, un programme par
# SM. Réglé à sec, jamais mesuré : ce sont les quatre boutons d'une seconde
# passe si le scellé (GEMM ≤ 80 ms) n'est pas atteint ; pas d'autotune, un
# banc au premier prefill fausserait la mesure de poste3.
_BN, _BK, _WARPS, _STAGES = 128, 64, 8, 2


def _programmes(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4                                              # interpréteur


def gemm_groupe(xs: torch.Tensor, w: torch.Tensor, tiles, m: Optional[int] = None) -> torch.Tensor:
    """``xs`` [G, K] bf16 trié par expert, ``w`` [E, M, K] bf16, ``tiles`` =
    ``(te, t0, tn)`` de ``MoEBlock._tuiles(cnt, BT)`` → ``y`` [G, M] bf16
    (``m`` colonnes valides si la pile est rembourrée : le reste n'est pas
    calculé). Une ligne hors de toute tuile (impossible si la grille couvre
    ``cnt``) resterait à zéro."""
    te, t0, tn = tiles
    G, K = xs.shape
    E, M_pile, K_w = w.shape
    assert K_w == K, (K_w, K)
    M = m or M_pile
    y = torch.zeros(G, M_pile, dtype=xs.dtype, device=xs.device)
    if G == 0 or te.numel() == 0:
        return y
    grille = (min(_programmes(xs.device), int(te.numel()) * -(-M // _BN)),)
    _gemm_groupe_kernel[grille](
        xs, w, y, te, t0, tn, te.numel(), M, K,
        xs.stride(0), w.stride(0), w.stride(1), y.stride(0),
        BT=BT, BN=_BN, BK=_BK, num_warps=_WARPS, num_stages=_STAGES)
    return y
