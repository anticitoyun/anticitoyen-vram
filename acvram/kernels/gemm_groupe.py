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


if triton is not None:

    @triton.jit
    def _e2m1(codes):
        """Quartet E2M1 (bit 3 = signe, bits 2-1 = exposant, bit 0 = mantisse)
        → fp32 : {0, 0,5, 1, 1,5, 2, 3, 4, 6}, par arithmétique (pas de
        table : l'interpréteur et le noyau font la même chose)."""
        mag = codes & 7
        mant = (mag & 1).to(tl.float32)
        expo = (mag >> 1).to(tl.float32)
        val = tl.where(expo == 0, mant * 0.5, (1.0 + 0.5 * mant) * tl.exp2(expo - 1.0))
        return tl.where((codes & 8) != 0, -val, val)

    @triton.jit
    def _e4m3(octets):
        """Octet E4M3 (fn) → fp32 par reconstruction des bits fp32 : normal
        (1 + m/8)·2^(e−7), sous-normal m·2^(−9), signe bit 7 — exact."""
        o = octets.to(tl.int32)
        signe = (o >> 7) & 1
        e = (o >> 3) & 15
        m = o & 7
        bits_norm = ((e - 7 + 127) << 23) | (m << 20)
        norm = tl.cast(bits_norm, tl.float32, bitcast=True)
        sous = m.to(tl.float32) * 0.001953125             # 2^-9
        val = tl.where(e == 0, sous, norm)
        return tl.where(signe != 0, -val, val)

    @triton.jit
    def _decode_kernel(src, dst, N: tl.constexpr, QUATRE: tl.constexpr):
        """Sonde : décode N octets/quartets pour vérifier les décodeurs contre torch."""
        i = tl.arange(0, N)
        o = tl.load(src + i)
        if QUATRE:
            v = _e2m1(o)
        else:
            v = _e4m3(o)
        tl.store(dst + i, v)

    @triton.jit
    def _gemm_groupe_nvfp4_kernel(xs_ptr, qw_ptr, bs_ptr, gs_ptr, y_ptr, te_ptr, t0_ptr, tn_ptr,
                                  n_tuiles, M, K,
                                  stride_xg, stride_qe, stride_qm, stride_be, stride_bm,
                                  stride_gse, stride_gsm, stride_yg,
                                  BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """B1 : même grille que `_gemm_groupe_kernel`, opérande B lu en NVFP4
        (E2M1 par paires, échelle E4M3 par bloc de 16, échelle globale par
        expert ou par ligne) et déquantifié en registres avant `tl.dot`.
        Les poids ne sont jamais écrits en bf16 : ~0,56 o/param lus au lieu
        de 2 + 2 (écriture puis relecture de la pile bf16)."""
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        n_blocs_n = tl.cdiv(M, BN)
        total = n_tuiles * n_blocs_n
        for t in range(pid, total, nprog):
            tuile = t // n_blocs_n
            nb = t % n_blocs_n
            e = tl.load(te_ptr + tuile)
            t0 = tl.load(t0_ptr + tuile)
            n = tl.load(tn_ptr + tuile)
            lignes = tl.arange(0, BT)
            masque_l = lignes < n
            rows = t0 + lignes
            cols = nb * BN + tl.arange(0, BN)
            masque_c = cols < M
            acc = tl.zeros((BT, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                masque_k = ks < K
                a = tl.load(xs_ptr + rows[:, None] * stride_xg + ks[None, :],
                            mask=masque_l[:, None] & masque_k[None, :], other=0.0)
                # octets de codes : colonne 2j (quartet bas) et 2j+1 (haut)
                kb = k0 // 2 + tl.arange(0, BK // 2)
                masque_kb = kb < K // 2
                oct = tl.load(qw_ptr + e * stride_qe + cols[:, None] * stride_qm + kb[None, :],
                              mask=masque_c[:, None] & masque_kb[None, :], other=0)
                w = tl.reshape(tl.join(_e2m1(oct & 15), _e2m1(oct >> 4)), (BN, BK))
                # échelles de bloc : une par 16 colonnes
                ksc = k0 // 16 + tl.arange(0, BK // 16)
                masque_sc = ksc < K // 16
                sc = tl.load(bs_ptr + e * stride_be + cols[:, None] * stride_bm + ksc[None, :],
                             mask=masque_c[:, None] & masque_sc[None, :], other=0)
                scf = tl.reshape(tl.broadcast_to(tl.expand_dims(_e4m3(sc), 2), (BN, BK // 16, 16)), (BN, BK))
                b = (w * scf).to(a.dtype)
                acc = tl.dot(a, tl.trans(b), acc)
            gs = tl.load(gs_ptr + e * stride_gse + cols * stride_gsm, mask=masque_c, other=0.0)
            acc = acc * gs[None, :]
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


def gemm_groupe_nvfp4(xs: torch.Tensor, qw: torch.Tensor, bs: torch.Tensor, gs: torch.Tensor,
                      tiles, m: Optional[int] = None) -> torch.Tensor:
    """B1 : ``xs`` [G, K] trié par expert ; pile NVFP4 ``qw`` [E, M, K_pile/2]
    uint8, ``bs`` [E, M, K_pile/16] octets E4M3, ``gs`` [E] fp32 (une échelle
    globale par expert) ou [E, M] (par ligne de sortie) → ``y`` [G, M].
    ``K`` est celui de ``xs`` : une pile rembourrée au-delà n'est pas lue."""
    te, t0, tn = tiles
    G, K = xs.shape
    E, M_pile = qw.shape[0], qw.shape[1]
    assert K % 16 == 0 and K <= qw.shape[2] * 2, (K, qw.shape)
    M = m or M_pile
    y = torch.zeros(G, M_pile, dtype=xs.dtype, device=xs.device)
    if G == 0 or te.numel() == 0:
        return y
    if gs.dim() == 1:
        stride_gse, stride_gsm = gs.stride(0), 0
    else:
        stride_gse, stride_gsm = gs.stride(0), gs.stride(1)
    grille = (min(_programmes(xs.device), int(te.numel()) * -(-M // _BN)),)
    _gemm_groupe_nvfp4_kernel[grille](
        xs, qw, bs, gs, y, te, t0, tn, te.numel(), M, K,
        xs.stride(0), qw.stride(0), qw.stride(1), bs.stride(0), bs.stride(1),
        stride_gse, stride_gsm, y.stride(0),
        BT=BT, BN=_BN, BK=_BK, num_warps=_WARPS, num_stages=_STAGES)
    return y


def decoder(octets: torch.Tensor, quatre: bool) -> torch.Tensor:
    """Les décodeurs du noyau, appliqués à un vecteur d'octets (sonde de test)."""
    n = octets.numel()
    assert n & (n - 1) == 0, "puissance de 2"
    out = torch.zeros(n, dtype=torch.float32, device=octets.device)
    _decode_kernel[(1,)](octets, out, N=n, QUATRE=quatre)
    return out


def tuiles_un_expert(n: int, device) -> tuple:
    """Grille (te, t0, tn) d'une projection non groupée : un seul « expert »,
    ``ceil(n / BT)`` tuiles, sans passer par MoEBlock."""
    nt = max(1, -(-n // BT))
    t0 = torch.arange(nt, device=device, dtype=torch.int32) * BT
    tn = (n - t0).clamp(0, BT).to(torch.int32)
    return torch.zeros(nt, dtype=torch.int32, device=device), t0, tn


def nvfp4_linear(x: torch.Tensor, t) -> torch.Tensor:
    """``x @ W.T`` pour une projection NVFP4 non groupée (`nvfp4_matmul`,
    régime ``ACVRAM_PREFILL=w4a16``) : le noyau B1 avec E = 1, échelle
    globale par ligne si le tenseur en porte (`global_scale_rows`, q/k/v
    fusionnés). Sortie dans le dtype de ``x``, calcul en bf16."""
    forme = x.shape
    xf = x.reshape(-1, forme[-1])
    dt = xf.dtype if xf.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    xf = xf.to(dt)
    if xf.shape[1] % 16:                                   # entrée plus courte que padded_in
        xf = torch.nn.functional.pad(xf, (0, 16 - xf.shape[1] % 16))
    xf = xf.contiguous()
    qw = t.qweight.unsqueeze(0)
    bs = t.block_scale.view(torch.uint8).unsqueeze(0)
    gsr = getattr(t, "global_scale_rows", None)
    gs = (gsr.to(torch.float32).reshape(1, -1) if gsr is not None
          else t.global_scale.to(torch.float32).reshape(1))
    m = t.shape[0]
    y = gemm_groupe_nvfp4(xf, qw, bs, gs.contiguous(), tuiles_un_expert(xf.shape[0], xf.device), m=m)
    return y[:, :m].to(x.dtype).reshape(*forme[:-1], m)
