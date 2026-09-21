"""Disposition Marlin des experts NVFP4, côté torch (C17, chantier-c17-mma2-lit-marlin-19-09).

Deux gestes, tous deux SANS carte :

* ``repack_torch`` (jumeau de ``ops.gptq_marlin_repack``, repris de la branche
  oceane-c1-w4a8 / C2) et ``_indices`` : la permutation des quartets d'une tuile
  16 k × 64 n — une voie ``t = 4·c + j`` du repack porte les colonnes
  ``w·16 + c`` et ``+8`` (w = 0..3, un mot par w) aux k ``2·j + {0, 1, 8, 9}``,
  quartets dans l'ordre ``_PACK_IDX`` (0 2 4 6 1 3 5 7 : n0:k, n0:k+8, n8:k,
  n8:k+8, n0:k+1, n0:k+9, n8:k+1, n8:k+9).
* ``fragments_b_depuis_marlin`` : EXACTEMENT l'arithmétique d'index du noyau
  ``nvfp4_gemm_grouped_mma2_kernel<…, MARLIN=true>`` (acvram_kernels.cu) —
  pour chaque colonne n et chaque paire de k, quel mot de quelle voie, quels
  quartets — rendue comme la pile naturelle ``[N, K/2]`` uint8 ; et
  ``echelles_ue4m3_depuis_marlin`` : la reconversion S0E5M3 → UE4M3 des octets
  d'échelle (``gm2_ue4m3_depuis_marlin``). Le test à sec les juge contre la pile
  d'origine : égalité stricte des quartets, égalité des échelles hors celles
  que le repack annule (s·facteur·2⁷ < 2).
"""
from __future__ import annotations

import math

import torch

_PACK_IDX = (0, 2, 4, 6, 1, 3, 5, 7)
_TC_OFFSETS = (0, 1, 8, 9)
_SWAP4 = (0, 2, 1, 3)
_INDICES: dict = {}


def _indices(device):
    """Indices fixes d'une tuile : ``repack`` [32, 4, 8] = source (kl·64 + nl) de
    chaque quartet du mot (t, w) ; ``echelles`` [64] = octet d'échelle de la colonne o."""
    cle = str(device)
    if cle in _INDICES:
        return _INDICES[cle]
    t = torch.arange(32).view(32, 1, 1)
    w = torch.arange(4).view(1, 4, 1)
    v = torch.tensor(_PACK_IDX).view(1, 1, 8)
    nl = w * 16 + t // 4 + 8 * (v // 4)
    kl = (t % 4) * 2 + torch.tensor(_TC_OFFSETS)[v % 4]
    repack = (kl * 64 + nl).to(torch.long)
    o = torch.arange(64)
    q = o // 8
    echelles = (8 * (o % 8) + ((q & ~3) | torch.tensor(_SWAP4)[q & 3])).to(torch.long)
    res = (repack.to(device), echelles.to(device))
    _INDICES[cle] = res
    return res


def repack_torch(q: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Jumeau torch de ``ops.gptq_marlin_repack(q, ∅, K, N, 4, False)`` : ``q``
    [K/8, N] int32 GPTQ (le quartet k aux bits 4·(k % 8) du mot k/8) →
    [K/16, 2N] int32, tuiles 16 × 64 de 128 mots."""
    assert tuple(q.shape) == (K // 8, N) and K % 64 == 0 and N % 64 == 0, (tuple(q.shape), K, N)
    idx, _ = _indices(q.device)
    dec = (torch.arange(8, device=q.device, dtype=torch.int32) * 4).view(1, 8, 1)
    codes = ((q.to(torch.int32).unsqueeze(1) >> dec) & 0xF).to(torch.uint8)           # [K/8, 8, N] = [K, N]
    tuiles = codes.view(K // 16, 16, N // 64, 64).permute(0, 2, 1, 3).reshape(K // 16, N // 64, 1024)
    quartets = tuiles[:, :, idx.view(-1)].view(K // 16, N // 64, 32, 4, 4, 2)          # [.., t, w, octet, moitié]
    octets = quartets[..., 0] | (quartets[..., 1] << 4)
    return octets.contiguous().view(torch.int32).view(K // 16, 2 * N)


def pile_naturelle_vers_marlin(qw: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """``qw`` [N, K/2] uint8 (paires E2M1, bas d'abord) → w_marlin [K/16, 2N] int32,
    par le même geste que ``preparer_pile`` (q = qw.view(int32).T : GPTQ, K empaqueté)."""
    q = qw.contiguous().view(torch.int32).T.contiguous()
    return repack_torch(q, K, N)


def fragments_b_depuis_marlin(w_marlin: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """L'arithmétique d'index du noyau MARLIN, en torch, mot par mot : pour la
    colonne n (o = n % 64, w = o/16, h = (o%16)/8, c = o%8) et la paire de k
    (2j, 2j+1) (+8 sous kb), l'octet j du mot de fragment vient du mot w de la
    voie t = 4c + j : quartet bas à la position pA = kb + 2h, quartet haut à
    pA + 4. Rend la pile naturelle reconstituée [N, K/2] uint8."""
    Kt, N2 = w_marlin.shape
    assert Kt == K // 16 and N2 == 2 * N
    octets = w_marlin.contiguous().view(torch.uint8).view(K // 16, N // 64, 32, 4, 4)    # [kt, nt, t, w, octet]
    mots = (octets[..., 0].to(torch.int64) | (octets[..., 1].to(torch.int64) << 8)
            | (octets[..., 2].to(torch.int64) << 16) | (octets[..., 3].to(torch.int64) << 24))  # [kt, nt, t, w]
    n = torch.arange(N, device=w_marlin.device)
    nt, o = n // 64, n % 64
    w, h, c = o // 16, (o % 16) // 8, o % 8
    out = torch.zeros(N, K // 2, dtype=torch.uint8, device=w_marlin.device)
    for kt in range(K // 16):
        for kb in (0, 1):                                            # moitié de 8 k de la tuile
            pA = kb + 2 * h                                          # [N]
            mot = torch.zeros(N, dtype=torch.int64, device=w_marlin.device)
            for j in range(4):
                W = mots[kt, nt, 4 * c + j, w]                       # [N]
                nl = (W >> (4 * pA)) & 0xF
                nh = (W >> (4 * (pA + 4))) & 0xF
                mot |= (nl | (nh << 4)) << (8 * j)
            for b in range(4):
                out[:, kt * 8 + kb * 4 + b] = ((mot >> (8 * b)) & 0xFF).to(torch.uint8)
    return out


def decal_echelles(facteur: float) -> int:
    """``decal`` du noyau : 15 + log2 facteur, soustrait au champ d'exposant (octet >> 3)
    — facteur est une puissance de 2 (biais half 15, biais UE4M3 7, ×2⁷ du repack)."""
    lf = int(round(math.log2(facteur)))
    assert 2.0 ** lf == facteur, facteur
    return 15 + lf


def ue4m3_depuis_marlin(octet: torch.Tensor, decal: int) -> torch.Tensor:
    """``gm2_ue4m3_depuis_marlin`` en torch : octet S0E5M3 (exposant half, 3 bits
    de mantisse) → octet UE4M3 ; 0 reste 0 ; exposant ≤ 0 → forme sous-normale."""
    b = octet.to(torch.int64)
    E = (b >> 3) - decal
    m = b & 7
    normal = (E << 3) | m
    sous = (8 | m) >> (1 - E).clamp(min=0)
    r = torch.where(E >= 1, normal, torch.where(E >= -2, sous, torch.zeros_like(b)))
    return torch.where(b == 0, torch.zeros_like(b), r).to(torch.uint8)


def echelles_ue4m3_depuis_marlin(s_marlin: torch.Tensor, decal: int, K: int, N: int) -> torch.Tensor:
    """s_marlin [K/16, N] octets (colonnes permutées par tuile de 64) → [N, K/16]
    UE4M3 dans l'ordre naturel, par l'index d'échelle du noyau."""
    _, ech = _indices(s_marlin.device)
    n = torch.arange(N, device=s_marlin.device)
    bidx = (n // 64) * 64 + ech[n % 64]                                  # octet de la colonne n dans sa ligne
    perm = s_marlin[:, bidx]                                             # [K/16, N] dans l'ordre naturel
    return ue4m3_depuis_marlin(perm, decal).T.contiguous()
