"""GEMM étroit W8A16 en Triton — poste C (sage-profil-verdict-17-09 § 2) :
les linéaires INT8 denses du décodage à b ≤ 16 (`narrow_gemm_kernel<32>`
2,08 ms pour 0,9 Go à 24 % de la bande passante, `int8_gemv` de la tête
0,88 ms pour 0,31 Go — Laure 6a4fd57).

Poids uint8 affine par groupes (`INT8Tensor` : q [N, K], échelle fp16 et
zéro uint8 [N, K/G]) : par groupe g, ``y += (x_g · q_gᵀ − Σx_g · z_g) · s_g``
— le produit se fait sur les entiers (uint8 ≤ 255 exact en bf16) par
``tl.dot`` en fp32, le zéro ne coûte qu'une somme de ligne, l'échelle un
produit par colonne. Grille (tuile N, tranche K) pour couvrir la carte à
b = 1 comme à b = 12 ; tranches réduites par une somme torch (déterministe).
Sortie bf16, ou fp32 pour la tête (logits).

Scellé (Sage) : dense b = 12 ≤ 1,0 ms par pas (−1,8 ms), PPL inchangée ;
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
    def _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, g0, groupes_par_tranche,
                     rows, cols, masque_m, masque_n, stride_xm, stride_qn, stride_sn,
                     BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        """Accumulateur fp32 [BM, BN] d'UNE tranche K (groupes g0 .. g0 + gpt) :
        le corps commun de `_etroit_kernel` (témoin, somme torch des tranches)
        et de `_etroit_reduit_kernel` (C15 niveau 3)."""
        kk = tl.arange(0, G)
        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        for g in range(g0, g0 + groupes_par_tranche):
            # la dernière tranche déborde quand NG n'est pas multiple de la
            # taille de tranche : g ≥ NG lisait l'échelle et le zéro AU-DELÀ
            # du tenseur (cols·NG + g), accès illégal sous graphe dans le
            # moteur (Laure b34a1bb) — invisible au banc, dont les tenseurs
            # voisins absorbaient la lecture
            masque_g = g < NG
            ks = g * G + kk
            masque_k = (ks < K) & masque_g
            x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :],
                        mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :],
                        mask=masque_n[:, None] & masque_k[None, :], other=0)
            s = tl.load(s_ptr + cols * stride_sn + g, mask=masque_n & masque_g, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + cols * stride_sn + g, mask=masque_n & masque_g, other=0).to(tl.float32)
            prod = tl.dot(x, tl.trans(q.to(x.dtype)))                     # [BM, BN] fp32
            sx = tl.sum(x.to(tl.float32), 1)                              # Σ_k x[m, k] du groupe
            acc += (prod - sx[:, None] * z[None, :]) * s[None, :]
        return acc


    @triton.jit
    def _etroit_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, M, N, K, NG, groupes_par_tranche,
                       stride_xm, stride_qn, stride_sn, stride_ys, stride_ym,
                       BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        acc = _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * groupes_par_tranche,
                           groupes_par_tranche, rows, cols, masque_m, masque_n,
                           stride_xm, stride_qn, stride_sn, BM_, BN_, G)
        tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                 acc, mask=masque_m[:, None] & masque_n[None, :])


    @triton.jit
    def _etroit_reduit_kernel(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, cnt_ptr, out_ptr, M, N, K, NG,
                              groupes_par_tranche, tranches,
                              stride_xm, stride_qn, stride_sn, stride_ys, stride_ym, stride_om,
                              BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr):
        """C15 niveau 3 : le GEMM étroit en UN nœud — chaque programme (tuile N,
        tranche K) écrit son accumulateur fp32 dans y[ps], incrémente le
        compteur de sa tuile ; le DERNIER arrivé somme les tranches dans
        l'ordre 0..tranches−1 (déterministe quel que soit l'arrivant), écrit la
        sortie dans le dtype de out (bf16 : un seul arrondi, comme
        `y.sum(0).to(bf16)` ; fp32 pour la tête) et remet le compteur à zéro.
        Remplace torch.zeros + noyau + y.sum(0) + cast (4 nœuds). Ordre de la
        somme : sériel, celui du témoin quand torch le fait sériel (≤ 4
        tranches) ; sinon ± 1 ulp bf16. Visibilité : barrière, atomique
        acq_rel portée gpu, barrière, lectures .cg."""
        pn = tl.program_id(0)
        ps = tl.program_id(1)
        rows = tl.arange(0, BM_)
        cols = pn * BN_ + tl.arange(0, BN_)
        masque_m = rows < M
        masque_n = cols < N
        masque = masque_m[:, None] & masque_n[None, :]
        acc = _acc_tranche(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * groupes_par_tranche,
                           groupes_par_tranche, rows, cols, masque_m, masque_n,
                           stride_xm, stride_qn, stride_sn, BM_, BN_, G)
        tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
        tl.debug_barrier()
        n = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if n == tranches - 1:
            somme = tl.zeros((BM_, BN_), dtype=tl.float32)
            for t in range(0, tranches):
                somme += tl.load(y_ptr + t * stride_ys + rows[:, None] * stride_ym + cols[None, :],
                                 mask=masque, other=0.0, cache_modifier=".cg")
            tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :],
                     somme.to(out_ptr.dtype.element_ty), mask=masque)
            tl.store(cnt_ptr + pn, 0)


def _programmes(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4


G_MAX = 128


def eligible(t) -> bool:
    """Le noyau étroit prend le GROUPE comme tuile K : un poids symétrique par
    canal (group_size = K, convertis -qkvo-i8c, P2) demanderait 393 Kio de
    mémoire partagée pour 101 Kio disponibles (OutOfResources en service,
    19/09, godet 2 et préfill chunké) — dimension « taille de groupe » jamais
    posée (MECANISMES). Refusé au-delà de G_MAX : l'appelant prend narrow_gemm
    CUDA ou le GEMV, sortie inchangée pour tout poids à groupes de 128."""
    return t.group_size <= G_MAX


_COMPTEURS: dict = {}


def _compteur(n: int, device) -> torch.Tensor:
    """Compteurs int32 [n] des tuiles N, nuls entre deux lancements (le dernier
    programme remet le sien à zéro) : réservés une fois par forme et appareil,
    jamais pendant une capture de graphe (l'échauffement eager les crée)."""
    cle = (n, str(device))
    c = _COMPTEURS.get(cle)
    if c is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("gemm_etroit : compteurs du noyau fusionné réservés pendant une capture "
                               "de graphe — l'échauffement eager doit précéder (REGLES § 7)")
        c = _COMPTEURS[cle] = torch.zeros(n, dtype=torch.int32, device=device)
    return c


def gemm_etroit(x: torch.Tensor, t, sortie_fp32: bool = False, compact: bool = False) -> torch.Tensor:
    """``x`` [M ≤ 16, K] bf16 (fp16 sous l'interpréteur), ``t`` INT8Tensor
    → [M, N] dans le dtype de x, ou fp32 (tête). ``compact`` (C15 niveau 3,
    ACVRAM_GLUE_COMPACT) : un seul lancement, tranches réduites par le
    dernier programme de chaque tuile, sortie écrite dans son dtype."""
    M, K = x.shape
    N, k_pad = t.qweight.shape
    G = t.group_size
    assert M <= BM and K <= k_pad and k_pad % G == 0, (M, K, k_pad, G)
    ng = k_pad // G
    assert t.scales.shape == (N, ng) and t.zeros.shape == (N, ng), (t.scales.shape, t.zeros.shape, N, ng)
    tuiles_n = -(-N // BN)
    voulu = -(-2 * _programmes(x.device) // tuiles_n)
    tranches = max(1, min(ng, voulu))
    gpt = -(-ng // tranches)
    tranches = -(-ng // gpt)
    if compact:
        y = torch.empty(tranches, M, N, dtype=torch.float32, device=x.device)
        out = torch.empty(M, N, dtype=torch.float32 if sortie_fp32 else x.dtype, device=x.device)
        _etroit_reduit_kernel[(tuiles_n, tranches)](
            x, t.qweight, t.scales, t.zeros, y, _compteur(tuiles_n, x.device), out, M, N, K, ng, gpt, tranches,
            x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1), out.stride(0),
            BM_=BM, BN_=BN, G=G, num_warps=_WARPS, num_stages=_STAGES)
        return out
    y = torch.zeros(tranches, M, N, dtype=torch.float32, device=x.device)
    _etroit_kernel[(tuiles_n, tranches)](
        x, t.qweight, t.scales, t.zeros, y, M, N, K, ng, gpt,
        x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1),
        BM_=BM, BN_=BN, G=G, num_warps=_WARPS, num_stages=_STAGES)
    out = y.sum(0) if tranches > 1 else y[0]
    return out if sortie_fp32 else out.to(x.dtype)
