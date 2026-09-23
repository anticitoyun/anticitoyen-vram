"""États récurrents à formes fixes, groupés par lots de créneaux.

Un créneau (`new_static` d'une couche GDN / KDA / Mamba2) est un jeu de VUES
dans un tampon groupé de `LOT` créneaux : les créneaux 0..b-1 (b ≤ LOT) sont
des tranches contiguës, et le décodage du lot par un noyau fla lit et écrit
les états en place, sans rassembler ni redistribuer (17/09, GDN d'abord).
`static_load` / `static_export` copient dans ces vues comme avant.
"""
from __future__ import annotations

import torch

LOT = 16


def nouveau_static(obj, device: torch.device, formes: dict[str, tuple]) -> dict:
    """``formes`` : {clé: forme d'UN créneau}. Rend {clé: vue, "lot": (n° lot, i)}."""
    lots = obj.__dict__.setdefault("_lots", [])
    n = obj.__dict__.setdefault("_n_statics", 0)
    if n % LOT == 0:
        lots.append({k: torch.zeros(LOT, *f, dtype=torch.float32, device=device)
                     for k, f in formes.items()})
    lot, i = lots[n // LOT], n % LOT
    obj.__dict__["_n_statics"] = n + 1
    st = {k: lot[k][i] for k in formes}
    st["lot"] = (n // LOT, i)
    return st


def tranches(obj, statics: list, b: int, cles: tuple) -> tuple[dict, bool]:
    """Les états des ``b`` premiers créneaux, groupés : (tenseurs [b, ...],
    contigu). Contigu = tranches du lot 0 sans copie ; sinon rassemblés par
    `stack` (l'appelant redistribue après le calcul)."""
    lots = obj.__dict__.get("_lots", [])
    contigu = (b <= LOT and bool(lots)
               and all(st.get("lot") == (0, i) for i, st in enumerate(statics[:b])))
    if contigu:
        return {k: lots[0][k][:b] for k in cles}, True
    return {k: torch.stack([st[k] for st in statics[:b]]) for k in cles}, False


def redistribuer(statics: list, b: int, groupes: dict) -> None:
    for i, st in enumerate(statics[:b]):
        for k, t in groupes.items():
            st[k].copy_(t[i])
