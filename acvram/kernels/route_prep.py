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


_INDEX: dict = {}


def index_jetons(t: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(tok [T·k], seq [T·k]) int32, réservés une fois par (T, k, appareil)."""
    cle = (t, k, str(device))
    if cle not in _INDEX:
        tok = torch.arange(t, device=device, dtype=torch.int32).repeat_interleave(k).contiguous()
        seq = torch.arange(t * k, device=device, dtype=torch.int32)
        _INDEX[cle] = (tok, seq)
    return _INDEX[cle]


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
