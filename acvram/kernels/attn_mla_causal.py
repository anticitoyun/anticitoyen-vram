"""Cœur d'attention MLA fusionné et causal au PRÉFILL, en Triton — C13-c forme 1
(chantier-c13c-19-09 § Correction poste7 ; poste7-fiches-c5b-c13c-c14b-20-09 § 2).

Ce que remplace ce noyau (`engine/mla.py`, chemin chunké du préfill) : par morceau
de 256 requêtes, `scores = einsum(q_eff, C) · scale` [256, nh, total] fp32 écrits
en HBM, `masked_fill` causal, `softmax`, `einsum(probs, C[:, :rank])` — 5 ops × 8
morceaux × 47 couches, et ≈ 70 Go/pas de scores écrits puis relus à L = 2 047
(chantier-c13c § chiffrage). Ici : UN lancement par couche, les scores ne
quittent jamais les registres, et la moitié masquée n'est pas calculée.

Forme (flash-attention sur le latent) : une tuile de BM requêtes × BN clés ;
softmax en ligne fp32 (m, l par ligne, exp2 avec log2 e) ; la boucle sur les
tuiles de clés s'arrête à la diagonale — les tuiles entièrement visibles
passent sans masque, seules les tuiles diagonales portent le masque causal
exact `pos_k ≤ passe + i`. Particularité MLA : K et V sont le MÊME tenseur
latent C [total, W] (V = C[:, :rank]) — la tuile de clés sert aux deux `tl.dot`.

Tuilage de W = rank + rope (GLM : 512 + 64 = 576 = 9 × 64) : `tl.arange` exige
une puissance de 2, donc la dimension de contraction de q·kᵀ est parcourue par
tranches de BK = 64 (W_TUILES = ⌈W/BK⌉, la dernière masquée si W % BK ≠ 0), et
p·v par tranches de BK sur rank seulement (RANK_TUILES = rank/BK, 8 sur GLM),
chacune avec son accumulateur [BM, BK] fp32 — l'accumulateur complet [BM, 512]
ne tient pas en un seul bloc de registres. Lectures : la tuile C [BN, W] est lue
une fois par tranche BK pour les scores, puis relue par tranche pour p·v ; la
seconde lecture touche les mêmes adresses que la première (tuile bf16 de 74 Ko,
< L1) — C entier (2,4 Mo bf16 par couche) reste en L2 pendant tout le
lancement : c'est une lecture HBM par couche, contre 8 aujourd'hui.

Forme 1 (ici) : opérandes fp32, `tl.dot` en précision IEEE (`input_precision=
"ieee"`, l'équivalent exact de l'ancien `allow_tf32=False` — le mot-clé est
déprécié en Triton 3.8 et, laissé implicite, cède à `TRITON_F32_DEFAULT`),
accumulation fp32, softmax fp32 : sortie = chemin einsum fp32 ± ulp d'ordre de
sommation (juge : ≤ 8 ulp fp32 de l'amplitude par ligne, comme C14 ; aucune
porte de PPL). Forme 2 (tf32 / bf16 des opérandes, règle des 2 048 clés) passe
par le paramètre `operandes` du lanceur — réservé, NON implémenté ici.

Sans carte : `TRITON_INTERPRET=1` (le conftest le pose quand
CUDA_VISIBLE_DEVICES est vide) ; juge : tests/test_mla_flash_causal_c13c.py.
"""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None

BK = 64                    # tranche de la dimension W (q·kᵀ) et de rank (p·v)
BM_DEFAUT = 64             # requêtes par tuile (acc = RANK_TUILES × [BM, BK] fp32 en registres)
BN_DEFAUT = 64             # clés par tuile
# Tuile par carte (poste2, 20/09, `sonde-tuiles.py` sur sm_120) : à 64-64-w8-s2 Triton demande 319 488 o de
# shared contre 101 376 max par bloc (sm_120 : 99 Ko en opt-in) → OutOfResources ; les tuiles lancées :
# 32-64-w4-s1 (la plus grosse), 32-32, 16-64, 16-32, 32-16. La tuile se choisit donc par la shared de la
# carte, jamais par un défaut aveugle ; « nommé » = la ligne de régime dit la tuile.
_TUILES_PAR_SHARED = (              # (shared max par bloc ≥, BM, BN, num_warps, num_stages)
    (200 * 1024, 64, 64, 8, 2),     # H100/B200 (228 Ko) : la tuile d'origine
    (96 * 1024, 32, 64, 4, 1),      # sm_120 (99 Ko) : mesuré, se lance
    (0, 16, 32, 4, 1),              # tout le reste : la plus petite lancée
)
_CHOIX = {}


def tuile_par_carte(device) -> tuple:
    """(BM, BN, num_warps, num_stages) selon `shared_memory_per_block_optin` de la carte ;
    ACVRAM_MLA_FLASH_TUILE=BM,BN,warps,stages force (bancs, diagnostic)."""
    forcee = os.environ.get("ACVRAM_MLA_FLASH_TUILE")
    if forcee:
        bm, bn, w, st = (int(v) for v in forcee.split(","))
        return bm, bn, w, st
    cle = str(device)
    if cle not in _CHOIX:
        shared = 0
        try:
            if torch.cuda.is_available() and torch.device(device).type == "cuda":
                shared = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
        except Exception:                                     # noqa: BLE001
            shared = 0
        _CHOIX[cle] = next(t[1:] for t in _TUILES_PAR_SHARED if shared >= t[0])
    return _CHOIX[cle]
RANK_TUILES_MAX = 8        # accumulateurs explicites : rank ≤ 8 × BK = 512 (GLM : 512)
LOG2E = 1.4426950408889634

# Forme 1 : opérandes fp32, produit IEEE — FAUSSE sur sm_120 (verdict-c13c-19-09 : tl.dot fp32 IEEE
# est émulé, × 26 et 27 ulp du fp64). Forme 2 (poste7-c13c-forme1-faux-forme2-tf32-20-09 § 2) : le
# MÊME noyau en TF32 (tensor cores), sous la règle des 2 048 clés vues, jugé contre l'einsum TF32
# qu'il remplace ; « tf32x3 » (3 passes TF32 ≈ fp32) réservé à la sonde § 3. bf16 = F, non écrit.
_PRECISION = {"fp32": "ieee", "tf32": "tf32", "tf32x3": "tf32x3"}
_FORME2 = ("bf16",)


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _v_tranche(c_ptr, keys, keys_ok, s_ct, r, cols, BK: tl.constexpr, MASQUE: tl.constexpr):
        """V_r = C[keys, r·BK : (r+1)·BK] en fp32 (la même tuile de clés que les scores)."""
        ptr = c_ptr + keys[:, None] * s_ct + (r * BK + cols)[None, :]
        if MASQUE:
            v = tl.load(ptr, mask=keys_ok[:, None], other=0.0)
        else:
            v = tl.load(ptr)
        return v.to(tl.float32)

    @triton.jit
    def _tuile(q_base, c_ptr, k0, lignes_ok, pos_q, total, s_ct, scale_log2,
               m, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
               W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               W_TUILES: tl.constexpr, RANK_TUILES: tl.constexpr,
               PRECISION: tl.constexpr, MASQUE: tl.constexpr):
        """Une tuile de BN clés [k0, k0+BN) contre les BM requêtes du programme :
        scores en registres, mise à jour en ligne de (m, l, acc). MASQUE : tuile
        diagonale (clés ≤ pos_q et < total) ; sinon toutes les clés sont visibles."""
        cols = tl.arange(0, BK)
        keys = k0 + tl.arange(0, BN)
        keys_ok = keys < total
        s = tl.zeros((BM, BN), tl.float32)
        for kd in tl.static_range(W_TUILES):
            dcols = kd * BK + cols
            ok_d = dcols < W
            q = tl.load(q_base + dcols[None, :], mask=lignes_ok[:, None] & ok_d[None, :], other=0.0)
            if MASQUE:
                k = tl.load(c_ptr + keys[:, None] * s_ct + dcols[None, :],
                            mask=keys_ok[:, None] & ok_d[None, :], other=0.0)
            else:
                k = tl.load(c_ptr + keys[:, None] * s_ct + dcols[None, :],
                            mask=ok_d[None, :], other=0.0)
            # forme 1 : fp32 plein — jamais TF32 implicite (le défaut de Triton pour f32 × f32)
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), acc=s, input_precision=PRECISION)
        s = s * scale_log2                                  # scale · log2 e : softmax en base 2
        if MASQUE:
            visible = keys_ok[None, :] & (keys[None, :] <= pos_q[:, None])
            s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_sur = tl.where(m_new == float("-inf"), 0.0, m_new)   # ligne sans clé visible : pas de NaN
        corr = tl.exp2(m - m_sur)
        p = tl.exp2(s - m_sur[:, None])
        l = l * corr + tl.sum(p, 1)
        acc0 = acc0 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 0, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 1:
            acc1 = acc1 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 1, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 2:
            acc2 = acc2 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 2, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 3:
            acc3 = acc3 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 3, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 4:
            acc4 = acc4 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 4, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 5:
            acc5 = acc5 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 5, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 6:
            acc6 = acc6 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 6, cols, BK, MASQUE), input_precision=PRECISION)
        if RANK_TUILES > 7:
            acc7 = acc7 * corr[:, None] + tl.dot(p, _v_tranche(c_ptr, keys, keys_ok, s_ct, 7, cols, BK, MASQUE), input_precision=PRECISION)
        return m_new, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7

    @triton.jit
    def _flash_causal_kernel(q_ptr, c_ptr, o_ptr, t, total, passe, scale_log2,
                             s_qt, s_qh, s_ct, s_ot, s_oh,
                             W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, W_TUILES: tl.constexpr,
                             RANK_TUILES: tl.constexpr, PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        q0 = pid_m * BM
        rows = q0 + tl.arange(0, BM)
        lignes_ok = rows < t
        pos_q = passe + rows                                # position absolue : la requête i voit les clés 0..passe+i
        q_base = q_ptr + rows[:, None] * s_qt + h * s_qh
        m = tl.full((BM,), float("-inf"), tl.float32)
        l = tl.zeros((BM,), tl.float32)
        acc0 = tl.zeros((BM, BK), tl.float32)
        acc1 = tl.zeros((BM, BK), tl.float32)
        acc2 = tl.zeros((BM, BK), tl.float32)
        acc3 = tl.zeros((BM, BK), tl.float32)
        acc4 = tl.zeros((BM, BK), tl.float32)
        acc5 = tl.zeros((BM, BK), tl.float32)
        acc6 = tl.zeros((BM, BK), tl.float32)
        acc7 = tl.zeros((BM, BK), tl.float32)
        # tuiles entièrement visibles par TOUTES les lignes du bloc : clés ≤ passe + q0
        # (la première ligne, la plus contrainte) ; n_plein · BN ≤ passe + q0 + 1 ≤ total
        n_plein = (passe + q0 + 1) // BN
        fin = tl.minimum(total, passe + q0 + BM)            # dernière clé vue par le bloc + 1
        for k0 in range(0, n_plein * BN, BN):
            m, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _tuile(
                q_base, c_ptr, k0, lignes_ok, pos_q, total, s_ct, scale_log2,
                m, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                W, BM, BN, BK, W_TUILES, RANK_TUILES, PRECISION, False)
        for k0 in range(n_plein * BN, fin, BN):             # tuiles diagonales : masque causal exact
            m, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _tuile(
                q_base, c_ptr, k0, lignes_ok, pos_q, total, s_ct, scale_log2,
                m, l, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                W, BM, BN, BK, W_TUILES, RANK_TUILES, PRECISION, True)
        cols = tl.arange(0, BK)
        o_base = o_ptr + rows[:, None] * s_ot + h * s_oh
        tl.store(o_base + (0 * BK + cols)[None, :], acc0 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 1:
            tl.store(o_base + (1 * BK + cols)[None, :], acc1 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 2:
            tl.store(o_base + (2 * BK + cols)[None, :], acc2 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 3:
            tl.store(o_base + (3 * BK + cols)[None, :], acc3 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 4:
            tl.store(o_base + (4 * BK + cols)[None, :], acc4 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 5:
            tl.store(o_base + (5 * BK + cols)[None, :], acc5 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 6:
            tl.store(o_base + (6 * BK + cols)[None, :], acc6 / l[:, None], mask=lignes_ok[:, None])
        if RANK_TUILES > 7:
            tl.store(o_base + (7 * BK + cols)[None, :], acc7 / l[:, None], mask=lignes_ok[:, None])


def attention_mla_causale(q_eff: torch.Tensor, C: torch.Tensor, passe: int, scale: float,
                          rank: int, operandes: str = "fp32",
                          BM: int | None = None, BN: int | None = None,
                          num_warps: int | None = None, num_stages: int | None = None) -> torch.Tensor:
    """``q_eff`` [t, nh, W] fp32 (W = rank + rope), ``C`` [total, W] cache latent
    (bf16 ou fp32, converti en fp32 dans la tuile), ``passe`` = total − t (la
    requête i voit les clés 0..passe+i) → ``o_lat`` [t, nh, rank] fp32.

    ``operandes`` : "fp32" (forme 1, IEEE : émulé sur sm_120, faux), "tf32" (forme 2 : tensor
    cores TF32, le défaut candidat sous ≤ 2 048 clés), "tf32x3" (sonde § 3). "bf16" : non écrit."""
    if operandes in _FORME2:
        raise NotImplementedError(f"attention_mla_causale : operandes={operandes!r} : bf16 (= bras F) non écrit")
    if operandes not in _PRECISION:
        raise ValueError(f"attention_mla_causale : operandes={operandes!r} : fp32 | tf32 | tf32x3 | bf16")
    t, nh, W = q_eff.shape
    total = C.shape[0]
    bm_c, bn_c, w_c, s_c = tuile_par_carte(q_eff.device)
    BM = bm_c if BM is None else BM; BN = bn_c if BN is None else BN
    num_warps = w_c if num_warps is None else num_warps; num_stages = s_c if num_stages is None else num_stages
    assert C.shape[1] == W and C.stride(1) == 1 and q_eff.stride(2) == 1, (q_eff.shape, q_eff.stride(), C.shape, C.stride())
    assert total == passe + t, f"passe={passe} : attendu total − t = {total - t}"
    assert rank % BK == 0 and 1 <= rank // BK <= RANK_TUILES_MAX and rank <= W, (rank, W)
    assert q_eff.dtype is torch.float32, q_eff.dtype
    o_lat = torch.empty(t, nh, rank, dtype=torch.float32, device=q_eff.device)
    grille = (triton.cdiv(t, BM), nh)
    _flash_causal_kernel[grille](
        q_eff, C, o_lat, t, total, passe, float(scale) * LOG2E,
        q_eff.stride(0), q_eff.stride(1), C.stride(0), o_lat.stride(0), o_lat.stride(1),
        W=W, BM=BM, BN=BN, BK=BK, W_TUILES=triton.cdiv(W, BK), RANK_TUILES=rank // BK,
        PRECISION=_PRECISION[operandes], num_warps=num_warps, num_stages=num_stages)
    return o_lat


def reference_fp32(q_eff: torch.Tensor, C: torch.Tensor, passe: int, scale: float, rank: int) -> torch.Tensor:
    """Le chemin einsum fp32 de `mla.py` (préfill chunké, morceaux de 256), tel quel :
    la référence du juge, à ± 8 ulp fp32 de l'amplitude par ligne."""
    t = q_eff.shape[0]
    total = C.shape[0]
    C32 = C.to(torch.float32)
    V32 = C32[:, :rank]
    pos_k = torch.arange(total, device=C.device)
    morceaux = []
    for d0 in range(0, t, 256):
        d1 = min(t, d0 + 256)
        sc = torch.einsum('thr,sr->ths', q_eff[d0:d1].to(torch.float32), C32) * scale
        pos_q = torch.arange(d0, d1, device=C.device).unsqueeze(-1) + passe
        sc = sc.masked_fill(pos_k > pos_q.unsqueeze(1), float('-inf'))
        morceaux.append(torch.einsum('ths,sr->thr', sc.softmax(dim=-1), V32))
    return torch.cat(morceaux)
