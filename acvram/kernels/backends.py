"""Registre des backends de calcul : qui sait multiplier quoi, où.

L'idée demandée au projet : une abstraction de tenseur quantifié au-dessus de
plusieurs jeux de noyaux, sans faire d'acvram un enrobage de llama.cpp ou de
vLLM. Chaque *backend* déclare le format de poids qu'il lit, le type de
périphérique où il tourne, et une priorité ; l'exécution demande simplement
« multiplie ``x`` par ce poids » et le registre choisit, une fois pour toutes
par ``(format, périphérique)`` — le chemin chaud ne paie qu'un accès de
dictionnaire, comme l'ancien ``if/elif``.

Backends livrés : les noyaux CUDA fusionnés (sm_86+), le chemin tensor cores
FP4 (sm_100+), le GEMV C AVX2 du processeur, et la référence PyTorch qui
fonctionne partout et ferme la liste. En ajouter un — CUTLASS, cuBLASLt,
AVX-512 — est un ``register()`` de plus, pas une chirurgie du moteur ; le
``doctor`` liste ce que la machine courante a retenu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

__all__ = ["Backend", "register", "resolve", "matmul", "dequant", "table"]


@dataclass
class Backend:
    name: str
    formats: tuple[str, ...]
    device_type: str                      # "cuda" | "cpu"
    priority: int                         # plus grand = essayé d'abord
    available: Callable[[torch.device], bool]
    matmul: Callable[..., Optional[torch.Tensor]]
    dequant: Optional[Callable[..., torch.Tensor]] = None
    note: str = ""


_REGISTRY: list[Backend] = []
_RESOLVED: dict[tuple[str, str], list[Backend]] = {}


def register(backend: Backend) -> None:
    _REGISTRY.append(backend)
    _RESOLVED.clear()


_MASQUES: set[str] = set()


def masquer(nom: str) -> None:
    """Retire un backend de la résolution (bissection : `acvram.regime.masquer`).

    La porte `available = get_extension() is not None` de `cuda-fusionne`
    cachait un chemin torch (`nvfp4_mm_w4a8`) qu'aucun masque de noyau ne
    voyait (verdict-diff-moe-prefill-w4a8-17-09) : ici le masque porte sur
    le backend entier, la porte passe dessous."""
    if nom not in {b.name for b in _REGISTRY}:
        raise KeyError(f"backend inconnu : {nom} (connus : {sorted(b.name for b in _REGISTRY)})")
    _MASQUES.add(nom)
    _RESOLVED.clear()


def noms() -> list[str]:
    return [b.name for b in _REGISTRY]


def resolve(fmt: str, device: torch.device) -> list[Backend]:
    """Les backends candidats pour ce format sur ce périphérique, les plus
    prioritaires d'abord. Mémoïsé — la disponibilité d'une machine ne change
    pas en cours d'exécution."""
    # Par périphérique précis, pas par type : deux cartes CUDA de générations
    # différentes n'ont pas les mêmes capacités (la 3080 Ti n'a pas de tensor
    # cores FP4, la 5090 si).
    key = (fmt, str(device))
    got = _RESOLVED.get(key)
    if got is None:
        got = sorted((b for b in _REGISTRY
                      if fmt in b.formats and b.device_type == device.type
                      and b.name not in _MASQUES and b.available(device)),
                     key=lambda b: -b.priority)
        _RESOLVED[key] = got
    return got


def matmul(x: torch.Tensor, w: Any) -> torch.Tensor:
    """``x @ W.T`` pour un poids quantifié, par le meilleur backend qui accepte.

    Un backend peut décliner un appel précis (forme non gérée, seuil de lot) en
    rendant None : le suivant de la liste prend le relais. La référence
    PyTorch, priorité zéro, ne décline jamais.
    """
    fmt = getattr(w, "format", "plain")
    dev = (w.qweight.device if hasattr(w, "qweight")
           else getattr(w, "weight", x).device)
    for b in resolve(fmt, dev):
        y = b.matmul(x, w)
        if y is not None:
            return y
    raise RuntimeError(f"aucun backend pour {fmt} sur {dev}")


def dequant(w: Any, dtype: torch.dtype) -> torch.Tensor:
    fmt = getattr(w, "format", "plain")
    dev = w.qweight.device if hasattr(w, "qweight") else torch.device("cpu")
    for b in resolve(fmt, dev):
        if b.dequant is not None:
            return b.dequant(w, dtype)
    raise RuntimeError(f"aucun backend de dequantification pour {fmt} sur {dev}")


def table() -> list[dict]:
    """Ce que cette machine a retenu, pour ``acvram doctor``."""
    rows = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices += [torch.device(f"cuda:{i}")
                    for i in range(torch.cuda.device_count())]
    fmts = sorted({f for b in _REGISTRY for f in b.formats})
    for dev in devices:
        for fmt in fmts:
            names = [b.name for b in resolve(fmt, dev)]
            if names:
                rows.append({"device": str(dev), "format": fmt,
                             "backends": names})
    return rows
