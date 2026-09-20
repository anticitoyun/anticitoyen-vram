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
                valid, usage: torch.Tensor, num_warps: int = 4):
    """``logits`` [T, E] → (topw fp32 [T, k], topi int32 [T, k], eid int32 [T·k])
    — `moe_route` + `route_prep` en un lancement (F2). ``num_warps`` : 4 = le
    témoin (défaut Triton) ; 1 = C15-3c (réductions intra-warp), même
    arithmétique, ordre des sommes fp32 de la disposition à 1 warp."""
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
        AVEC_BIAIS=bias is not None and bias.numel() > 0, AVEC_VALID=valid is not None, num_warps=num_warps)
    return topw, topi, eid


def route_logits_fusee(x: torch.Tensor, w: torch.Tensor, bias, k: int, sigmoide: bool, renorm: bool,
                       scale: float, valid, usage: torch.Tensor, arrondi_bf16: bool = True):
    """C15-3c : le routeur compact = les logits par le MÊME appel cuBLAS que
    `_router_logits` (F.linear sur ``w`` — bf16 pour Coder, fp32 pour GLM :
    logits identiques au bit au témoin, donc routage identique) puis
    `_route_fusee_kernel` lancé à num_warps=1 : un jeton par programme, les
    128 experts dans UN warp — les 3 × k réductions de la sélection deviennent
    des shuffles intra-warp au lieu de passer par la mémoire partagée et des
    barrières (route_fusee : 7 µs/appel pour 12 jetons, verdict du 19/09).
    Même nombre de lancements que le témoin (wmma + splitKreduce + sélection :
    cuBLAS décide seul du split-K) : le gain visé est le temps de la sélection.
    ``arrondi_bf16`` : dtype des logits = celui de ``w`` (comme `_router_logits`).
    Les versions 0ca85673 (1 programme, 21,5 µs) et 3e62a9b5 (32 programmes,
    13 µs, latence d'un bloc par SM) calculaient les logits en Triton — routage
    ≠ témoin par construction (la sortie bf16 du témoin rend toute différence
    d'ordre de somme fp32 visible à 2⁻⁸ : |Δtopw| 10⁻³, experts qui basculent,
    verdict-c15-niveau3-coder-19-09 addendum 03 h 17) : retirées."""
    T, H = x.shape
    E, H2 = w.shape
    assert H == H2 and E <= 1024 and k <= 32, (x.shape, w.shape, k)
    logits = torch.nn.functional.linear(x.to(w.dtype), w)
    return route_fusee(logits, bias, k, sigmoide, renorm, scale, valid, usage, num_warps=1)
