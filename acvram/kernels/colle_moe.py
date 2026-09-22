"""Colle du préfill MoE en deux lancements (P0, sage-profil-verdict-18-09 § 1
(ii) : « colle MoE 8,3 ms dont radix-sort 384 appels ») : le tri des paires
(expert, jeton) par expert et la grille de tuiles de la GEMM groupée.

1. `trier_paires` : UN programme trie les G = T·top_k clés composites
   ``e · 2¹⁶ + paire`` par `tl.sort` (bitonique, déterministe — l'ordre
   stable d'`argsort(stable=True)` retombe de la clé), rend ``ordre`` (la
   paire de chaque rang), ``e_tri`` et l'histogramme ``cnt`` [E] par
   `tl.histogram`. Remplace argsort (radix, 8 lancements) + bincount.
   G ≤ 32 768 (Coder 2 048 × 8 = 16 384 ; au-delà : torch).
2. `tuiles` : depuis ``cnt``, en UN programme, la grille (e, t0, n) de
   `MoEBlock._tuiles` à taille fixe t_max — cumsum sur E puis pour chaque
   tuile l'expert par comparaison [t_max, E] (searchsorted) ; mêmes valeurs
   que la version torch (juge : tests/test_colle_moe.py).

Sortie de la GEMM inchangée : chaque ligne ne dépend que de son expert, pas
de sa place dans la tuile ; l'ordre rendu est de toute façon le même que
torch (clé composite = tri stable).
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

G_MAX = 32768


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _tri_kernel(e_ptr, ordre_ptr, etri_ptr, cnt_ptr, G, BG: tl.constexpr, E: tl.constexpr):
        i = tl.arange(0, BG)
        masque = i < G
        e = tl.load(e_ptr + i, mask=masque, other=E).to(tl.int32)       # au-delà de G : expert E (en queue)
        cle = e * 65536 + i
        tri = tl.sort(cle)
        rang_valide = i < G
        tl.store(ordre_ptr + i, tri & 65535, mask=rang_valide)
        tl.store(etri_ptr + i, tri >> 16, mask=rang_valide)
        h = tl.histogram(tl.where(masque, e, 0), E)
        # la sentinelle E est hors histogramme (E cases) ; les rembourrages
        # comptés en 0 sont retirés
        h0 = tl.sum(tl.where(masque, 0, 1), 0)
        j = tl.arange(0, E)
        tl.store(cnt_ptr + j, tl.where(j == 0, h - h0, h))

    # Un programme par bloc de BT_BLOC créneaux (T4 20/09, Jérôme : à bt = 16, t_max = 1 152 → BT_MAX = 2 048
    # constexpr et un tenseur [2 048, 128] déroulé dans UN programme : make_llir ne finissait jamais, 9 min à 104 %
    # CPU). Chaque programme refait les deux cumsum sur E (128 valeurs : gratuit) et sert ses BT_BLOC créneaux.
    BT_BLOC = 256

    @triton.jit
    def _tuiles_kernel(cnt_ptr, te_ptr, t0_ptr, tn_ptr, t_max, bt,
                       BE: tl.constexpr, BT_BLOC: tl.constexpr):
        j = tl.arange(0, BE)                       # experts (E ≤ BE)
        cnt = tl.load(cnt_ptr + j)                  # [BE] (E = BE exigé)
        starts = tl.cumsum(cnt, 0) - cnt
        ntiles = (cnt + bt - 1) // bt
        base = tl.cumsum(ntiles, 0) - ntiles
        s = tl.program_id(0) * BT_BLOC + tl.arange(0, BT_BLOC)   # créneaux de tuiles de ce programme
        masque_s = s < t_max
        # searchsorted(base, s, right=True) - 1 : nombre de base ≤ s, moins un
        n_le = tl.sum((base[None, :] <= s[:, None]).to(tl.int32), 1)
        e = tl.minimum(tl.maximum(n_le - 1, 0), BE - 1)
        # base[e], starts[e], cnt[e] par sélection (E petit : somme masquée)
        sel = (j[None, :] == e[:, None]).to(tl.int32)
        base_e = tl.sum(sel * base[None, :], 1)
        start_e = tl.sum(sel * starts[None, :], 1)
        cnt_e = tl.sum(sel * cnt[None, :], 1)
        idx = s - base_e
        t0 = start_e + idx * bt
        n = tl.minimum(tl.maximum(cnt_e - idx * bt, 0), bt)
        tl.store(te_ptr + s, e.to(tl.int32), mask=masque_s)
        tl.store(t0_ptr + s, t0.to(tl.int32), mask=masque_s)
        tl.store(tn_ptr + s, n.to(tl.int32), mask=masque_s)


def trier_paires(flat_e: torch.Tensor, E: int):
    """``flat_e`` [G] int (expert de chaque paire, ordre des jetons) →
    (ordre int64 [G], e_tri int64 [G], cnt int64 [E]) — les mêmes que
    argsort(stable) / flat_e[ordre] / bincount."""
    G = flat_e.numel()
    assert G <= G_MAX and E <= 65536, (G, E)
    BG = 1
    while BG < G:
        BG *= 2
    BG = max(BG, 16)
    e32 = flat_e.to(torch.int32).contiguous()
    ordre = torch.empty(G, dtype=torch.int32, device=flat_e.device)
    e_tri = torch.empty(G, dtype=torch.int32, device=flat_e.device)
    cnt = torch.empty(E, dtype=torch.int32, device=flat_e.device)
    _tri_kernel[(1,)](e32, ordre, e_tri, cnt, G, BG=BG, E=E, num_warps=min(32, max(4, BG // 512)))
    return ordre.to(torch.int64), e_tri.to(torch.int64), cnt.to(torch.int64)


def tuiles(cnt: torch.Tensor, bt: int, t_max: int):
    """La grille (tile_e, t0, n) int32 de `MoEBlock._tuiles(cnt, bt, t_max)`."""
    E = cnt.numel()
    assert E & (E - 1) == 0, "E puissance de 2 (128, 64, 256…)"
    te = torch.empty(t_max, dtype=torch.int32, device=cnt.device)
    t0 = torch.empty_like(te)
    tn = torch.empty_like(te)
    grille = (max(1, -(-t_max // BT_BLOC)),)
    _tuiles_kernel[grille](cnt.to(torch.int32).contiguous(), te, t0, tn, t_max, bt,
                           BE=E, BT_BLOC=BT_BLOC, num_warps=4)
    return te, t0, tn
