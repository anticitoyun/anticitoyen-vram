"""Préparation du routage MoE en UN lancement — poste F, fusion (1)
(verdict-lancements-b1-17-09 : à b = 1, 1 275 lancements par pas dont 380 de
colle torch autour de `_forward_grouped` — `masked_fill(~valid)`,
`_compter_routage` (copie int64, comparaison, copie, clamp, scatter_add),
`arange` × 2, `repeat_interleave`, copies — 0,33 ms sur 3,50).

Un noyau Triton lit `topi` [T, k] int32 et `valid` [T] bool et écrit
`eid` [T·k] int32 (−1 sur les créneaux fantômes) tout en incrémentant le
compteur d'usage des experts (atomiques int64 : des entiers, donc exact et
déterministe, même résultat que `scatter_add_`). Les index de jetons
(`tok` = arange(T) répété k fois, `seq` = arange(T·k)) ne dépendent que du
godet : ils sont réservés une fois par forme et réutilisés — sous graphe, le
godet est figé à la capture.

Aucune valeur ne change : `eid`, `tok`, `seq` et le compteur sont ceux du
chemin torch au bit près (juge : tests/test_route_prep.py).
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


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _route_prep_kernel(topi_ptr, valid_ptr, eid_ptr, usage_ptr, n,
                           K: tl.constexpr, BLOC: tl.constexpr, AVEC_VALID: tl.constexpr):
        i = tl.program_id(0) * BLOC + tl.arange(0, BLOC)
        masque = i < n
        e = tl.load(topi_ptr + i, mask=masque, other=-1)
        if AVEC_VALID:
            v = tl.load(valid_ptr + i // K, mask=masque, other=0)
            e = tl.where(v != 0, e, -1)
        tl.store(eid_ptr + i, e, mask=masque)
        reel = masque & (e >= 0)
        tl.atomic_add(usage_ptr + tl.where(reel, e, 0), 1, mask=reel)


    @triton.jit
    def _route_fusee_kernel(logits_ptr, bias_ptr, valid_ptr, topw_ptr, topi_ptr, eid_ptr, usage_ptr,
                            E, scale,
                            K: tl.constexpr, BE: tl.constexpr, SIGMOIDE: tl.constexpr,
                            RENORM: tl.constexpr, AVEC_BIAIS: tl.constexpr, AVEC_VALID: tl.constexpr):
        """F2 : `moe_route` (acvram_kernels.cu, même arithmétique : probabilités
        fp32, sélection sur probs + biais, égalités vers l'indice le plus bas,
        renormalisation 1/somme × échelle) et `route_prep` en un programme par
        jeton."""
        t = tl.program_id(0)
        i = tl.arange(0, BE)
        masque = i < E
        lg = tl.load(logits_ptr + t * E + i, mask=masque, other=float("-inf")).to(tl.float32)
        if SIGMOIDE:
            probs = 1.0 / (1.0 + tl.exp(-lg))
        else:
            m = tl.max(lg, 0)
            ex = tl.exp(lg - m)
            probs = ex / tl.sum(ex, 0)
        probs = tl.where(masque, probs, 0.0)
        if AVEC_BIAIS:
            sel = probs + tl.load(bias_ptr + i, mask=masque, other=0.0).to(tl.float32)
        else:
            sel = probs
        sel = tl.where(masque, sel, float("-inf"))
        if AVEC_VALID:
            v = tl.load(valid_ptr + t)
        else:
            v = 1
        jj = tl.arange(0, 32)
        mj = jj < K
        pw = tl.zeros((32,), dtype=tl.float32)                  # les k probabilités choisies, en registres
        somme = 0.0
        for j in range(K):
            bv = tl.max(sel, 0)
            bi = tl.min(tl.where(sel == bv, i, BE), 0)          # égalité : indice le plus bas
            pj = tl.sum(tl.where(i == bi, probs, 0.0), 0)
            somme += pj
            pw = tl.where(jj == j, pj, pw)
            tl.store(topi_ptr + t * K + j, bi.to(tl.int32))
            e = tl.where(v != 0, bi, -1)
            tl.store(eid_ptr + t * K + j, e.to(tl.int32))
            tl.atomic_add(usage_ptr + tl.where(v != 0, bi, 0), 1, mask=v != 0)
            sel = tl.where(i == bi, float("-inf"), sel)
        if RENORM:
            f = (1.0 / somme) * scale
        else:
            f = 1.0 * scale
        tl.store(topw_ptr + t * K + jj, pw * f, mask=mj)


    @triton.jit
    def _route_logits_fusee_kernel(x_ptr, w_ptr, part_ptr, cnt_ptr, bias_ptr, valid_ptr, topw_ptr, topi_ptr,
                                   eid_ptr, usage_ptr, T, E, H, scale, stride_xt, stride_we, NB, KS,
                                   K: tl.constexpr, BT: tl.constexpr, BE: tl.constexpr, BEB: tl.constexpr,
                                   BK: tl.constexpr, BH: tl.constexpr,
                                   SIGMOIDE: tl.constexpr, RENORM: tl.constexpr, AVEC_BIAIS: tl.constexpr,
                                   AVEC_VALID: tl.constexpr, ARRONDI_BF16: tl.constexpr):
        """C15 niveau 3 / 3b : les logits du routeur (x [T, H] bf16 · Wᵀ [H, E])
        DANS le noyau de sélection, à la place de F.linear (cuBLAS) puis
        `_route_fusee_kernel`. Grille (bloc de BT jetons, bloc de BEB experts,
        tranche de BK canaux) : chaque programme calcule un partiel fp32
        [BT, BEB] par `tl.dot` (produits exacts, accumulation fp32) et l'écrit
        dans part[pk, t, e] — 4 × 8 = 32 programmes pour Coder (E 128, H 2 048),
        W lu une fois en tout ; le DERNIER programme du bloc de jetons (compteur
        atomique acq_rel entre deux barrières, auto-remis à zéro) somme les KS
        partiels dans l'ordre 0..KS−1 (déterministe), arrondit en bf16 si le
        témoin sortait des logits bf16 (Coder : softmax sans biais ; fp32 sinon,
        GLM) et fait la sélection : même arithmétique que `_route_fusee_kernel`
        vectorisée sur les lignes. Seul l'ordre de la somme diffère du témoin :
        logits ± 1 ulp bf16, même sélection hors égalité à l'ulp. Version 3b :
        la version 0ca85673 (un seul programme, 32 itérations sérielles sur W)
        coûtait 21,5 µs (verdict-c15-niveau3-coder-19-09) ; visée ≤ 3 µs."""
        pt = tl.program_id(0)
        pn = tl.program_id(1)
        pk = tl.program_id(2)
        rows = pt * BT + tl.arange(0, BT)
        masque_t = rows < T
        colsb = pn * BEB + tl.arange(0, BEB)
        masque_b = colsb < E
        hh = tl.arange(0, BH)
        acc = tl.zeros((BT, BEB), dtype=tl.float32)
        for h0 in range(pk * BK, (pk + 1) * BK, BH):
            hs = h0 + hh
            mh = hs < H
            x = tl.load(x_ptr + rows[:, None] * stride_xt + hs[None, :],
                        mask=masque_t[:, None] & mh[None, :], other=0.0)
            w = tl.load(w_ptr + colsb[:, None] * stride_we + hs[None, :],
                        mask=masque_b[:, None] & mh[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w.to(x.dtype)))                  # [BT, BEB] fp32
        tl.store(part_ptr + (pk * T + rows[:, None]) * E + colsb[None, :], acc,
                 mask=masque_t[:, None] & masque_b[None, :])
        tl.debug_barrier()
        n = tl.atomic_add(cnt_ptr + pt, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if n == NB * KS - 1:
            i = tl.arange(0, BE)
            masque = i < E
            lg = tl.zeros((BT, BE), dtype=tl.float32)
            for q in range(0, KS):
                lg += tl.load(part_ptr + (q * T + rows[:, None]) * E + i[None, :],
                              mask=masque_t[:, None] & masque[None, :], other=0.0, cache_modifier=".cg")
            if ARRONDI_BF16:
                lg = lg.to(tl.bfloat16).to(tl.float32)
            lg = tl.where(masque[None, :], lg, float("-inf"))
            if SIGMOIDE:
                probs = 1.0 / (1.0 + tl.exp(-lg))
            else:
                m = tl.max(lg, 1)
                ex = tl.exp(lg - m[:, None])
                probs = ex / tl.sum(ex, 1)[:, None]
            probs = tl.where(masque[None, :], probs, 0.0)
            if AVEC_BIAIS:
                sel = probs + tl.load(bias_ptr + i, mask=masque, other=0.0).to(tl.float32)[None, :]
            else:
                sel = probs
            sel = tl.where(masque[None, :], sel, float("-inf"))
            if AVEC_VALID:
                v = tl.load(valid_ptr + rows, mask=masque_t, other=0)
                ok = masque_t & (v != 0)
            else:
                ok = masque_t
            jj = tl.arange(0, 32)
            mj = jj < K
            pw = tl.zeros((BT, 32), dtype=tl.float32)
            somme = tl.zeros((BT,), dtype=tl.float32)
            for j in range(K):
                bv = tl.max(sel, 1)
                bi = tl.min(tl.where(sel == bv[:, None], i[None, :], BE), 1)   # égalité : indice le plus bas
                pj = tl.sum(tl.where(i[None, :] == bi[:, None], probs, 0.0), 1)
                somme += pj
                pw = tl.where(jj[None, :] == j, pj[:, None], pw)
                tl.store(topi_ptr + rows * K + j, bi.to(tl.int32), mask=masque_t)
                e = tl.where(ok, bi, -1)
                tl.store(eid_ptr + rows * K + j, e.to(tl.int32), mask=masque_t)
                tl.atomic_add(usage_ptr + tl.where(ok, bi, 0), 1, mask=ok)
                sel = tl.where(i[None, :] == bi[:, None], float("-inf"), sel)
            if RENORM:
                f = (1.0 / somme) * scale
            else:
                f = somme * 0.0 + scale
            tl.store(topw_ptr + rows[:, None] * K + jj[None, :], pw * f[:, None],
                     mask=masque_t[:, None] & mj[None, :])
            tl.store(cnt_ptr + pt, 0)


_INDEX: dict = {}


def index_jetons(t: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(tok [T·k], seq [T·k]) int32, réservés une fois par (T, k, appareil)."""
    cle = (t, k, str(device))
    if cle not in _INDEX:
        tok = torch.arange(t, device=device, dtype=torch.int32).repeat_interleave(k).contiguous()
        seq = torch.arange(t * k, device=device, dtype=torch.int32)
        _INDEX[cle] = (tok, seq)
    return _INDEX[cle]


_INDEX_LONG: dict = {}


def index_jetons_long(t: int, k: int, device) -> torch.Tensor:
    """``tok`` [T·k] en int64, réservé une fois par (T, k, appareil) — C15 :
    l'indexation AWQ (``x[tok]``) le convertissait à chaque couche et chaque pas."""
    cle = (t, k, str(device))
    if cle not in _INDEX_LONG:
        _INDEX_LONG[cle] = index_jetons(t, k, device)[0].long().contiguous()
    return _INDEX_LONG[cle]


def route_prep(topi: torch.Tensor, valid, usage: torch.Tensor) -> torch.Tensor:
    """``topi`` [T, k] int32, ``valid`` [T] bool ou None, ``usage`` [E] int64
    (incrémenté en place) → ``eid`` [T·k] int32."""
    t, k = topi.shape
    n = t * k
    eid = torch.empty(n, dtype=torch.int32, device=topi.device)
    if n == 0:
        return eid
    topi_c = topi.contiguous()
    v = valid.contiguous() if valid is not None else eid          # pointeur ignoré sans valid
    BLOC = 256
    _route_prep_kernel[(-(-n // BLOC),)](
        topi_c, v, eid, usage, n, K=k, BLOC=BLOC, AVEC_VALID=valid is not None)
    return eid


def route_fusee(logits: torch.Tensor, bias, k: int, sigmoide: bool, renorm: bool, scale: float,
                valid, usage: torch.Tensor):
    """``logits`` [T, E] → (topw fp32 [T, k], topi int32 [T, k], eid int32 [T·k])
    — `moe_route` + `route_prep` en un lancement (F2)."""
    T, E = logits.shape
    assert E <= 1024 and k <= 32, (E, k)
    BE = 1
    while BE < E:
        BE *= 2
    topw = torch.empty(T, k, dtype=torch.float32, device=logits.device)
    topi = torch.empty(T, k, dtype=torch.int32, device=logits.device)
    eid = torch.empty(T * k, dtype=torch.int32, device=logits.device)
    if T == 0:
        return topw, topi, eid
    lg = logits.contiguous()
    b = bias.contiguous() if bias is not None and bias.numel() else lg
    v = valid.contiguous() if valid is not None else eid
    _route_fusee_kernel[(T,)](
        lg, b, v, topw, topi, eid, usage, E, float(scale),
        K=k, BE=max(BE, 32), SIGMOIDE=sigmoide, RENORM=renorm,
        AVEC_BIAIS=bias is not None and bias.numel() > 0, AVEC_VALID=valid is not None)
    return topw, topi, eid


BT_LOGITS = 16             # jetons par bloc (tl.dot : M ≥ 16) ; le godet ≤ 32 → 1-2 blocs
BEB_LOGITS = 32            # experts par programme
BK_LOGITS = 256            # canaux par programme : Coder (E 128, H 2 048) → 4 × 8 = 32 programmes
BH_LOGITS = 64             # pas de la boucle interne (tl.dot K ≥ 16)

_COMPTEURS: dict = {}


def _compteur(n: int, device) -> torch.Tensor:
    """Compteurs int32 [n] des blocs de jetons, nuls entre deux lancements (le
    dernier programme remet le sien à zéro) : réservés une fois par forme et
    appareil, jamais pendant une capture de graphe."""
    cle = (n, str(device))
    c = _COMPTEURS.get(cle)
    if c is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("route_prep : compteurs du routeur fusionné réservés pendant une capture "
                               "de graphe — l'échauffement eager doit précéder (REGLES § 7)")
        c = _COMPTEURS[cle] = torch.zeros(n, dtype=torch.int32, device=device)
    return c


def route_logits_fusee(x: torch.Tensor, w: torch.Tensor, bias, k: int, sigmoide: bool, renorm: bool,
                       scale: float, valid, usage: torch.Tensor, arrondi_bf16: bool = True):
    """C15 niveau 3 : ``x`` [T, H] bf16 (fp16 sous l'interpréteur), ``w`` [E, H]
    (poids du routeur, même dtype que x) → (topw fp32 [T, k], topi int32 [T, k],
    eid int32 [T·k]) en UN lancement : logits (partiels par blocs d'experts et
    tranches de canaux, sommés par le dernier programme, arrondis bf16 si
    ``arrondi_bf16`` comme la sortie de F.linear bf16) + `route_fusee`.
    Témoin : ``_router_logits`` puis `route_fusee` (ACVRAM_GLUE_COMPACT=0)."""
    T, H = x.shape
    E, H2 = w.shape
    assert H == H2 and E <= 1024 and k <= 32 and x.dtype == w.dtype, (x.shape, w.shape, k, x.dtype, w.dtype)
    BE = 1
    while BE < E:
        BE *= 2
    topw = torch.empty(T, k, dtype=torch.float32, device=x.device)
    topi = torch.empty(T, k, dtype=torch.int32, device=x.device)
    eid = torch.empty(T * k, dtype=torch.int32, device=x.device)
    if T == 0:
        return topw, topi, eid
    xc = x if x.stride(1) == 1 else x.contiguous()
    wc = w if w.stride(1) == 1 else w.contiguous()
    b = bias.contiguous() if bias is not None and bias.numel() else xc
    v = valid.contiguous() if valid is not None else eid
    nt = -(-T // BT_LOGITS)
    nb = -(-E // BEB_LOGITS)
    ks = -(-H // BK_LOGITS)
    part = torch.empty(ks, T, E, dtype=torch.float32, device=x.device)
    _route_logits_fusee_kernel[(nt, nb, ks)](
        xc, wc, part, _compteur(nt, x.device), b, v, topw, topi, eid, usage, T, E, H, float(scale),
        xc.stride(0), wc.stride(0), nb, ks,
        K=k, BT=BT_LOGITS, BE=max(BE, 32), BEB=BEB_LOGITS, BK=BK_LOGITS, BH=BH_LOGITS,
        SIGMOIDE=sigmoide, RENORM=renorm,
        AVEC_BIAIS=bias is not None and bias.numel() > 0, AVEC_VALID=valid is not None,
        ARRONDI_BF16=arrondi_bf16, num_warps=4)
    return topw, topi, eid
