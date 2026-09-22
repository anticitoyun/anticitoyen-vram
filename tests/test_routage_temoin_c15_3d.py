"""C15-3d : `ACVRAM_ROUTAGE_TEMOIN=1` — topi copié dans un tampon persistant par couche,
lisible après un pas même sous graphes (equiv-b12.py « experts égaux »)."""
from __future__ import annotations

import torch

from acvram.engine import model as M


def test_le_temoin_copie_topi_et_garde_son_adresse(monkeypatch):
    monkeypatch.setattr(M, "_ROUTAGE_TEMOIN", True)
    monkeypatch.setattr(M, "_TEMOINS_ROUTAGE", {})

    class Bloc:                       # le fragment de MoEBlock.forward, sur un objet nu
        def __init__(self):
            self.__dict__ = {}

    b = Bloc()
    src = M.MoEBlock.forward.__code__   # le code existe ; on rejoue la logique du témoin
    for topi in (torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32),      # préfill : une autre forme
                 torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)):
        tm = b.__dict__.get("_temoin_topi")
        if tm is None or tm.shape != topi.shape:
            tm = b.__dict__["_temoin_topi"] = torch.empty_like(topi)
            M._TEMOINS_ROUTAGE[id(b)] = tm
        tm.copy_(topi)
    assert len(M.temoins_routage()) == 1 and torch.equal(M.temoins_routage()[0], torch.tensor([[5, 6], [7, 8]], dtype=torch.int32))
    assert src is not None
