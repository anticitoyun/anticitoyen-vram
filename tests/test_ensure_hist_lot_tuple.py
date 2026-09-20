"""C3 MTP sur hybride GDN (verdict-mtp-exact-19-09) : `static["lot"]` est un tuple
(lot_etats.nouveau_static) — `ensure_hist` mourait sur `.shape` au premier pas
spéculatif ; seuls les tenseurs d'état ont un historique, `rollback` ne touche qu'eux."""
from __future__ import annotations

import types

import torch

from acvram.engine import model as M


def _couche():
    c = types.SimpleNamespace(static_hist=None, static_hist_len=0, index=3,
                              static={"conv": torch.ones(4, 3), "S": torch.full((1, 2, 2), 2.0), "lot": (0, 1),
                                      "cache": torch.zeros(8, 4), "scores": torch.zeros(2, 8)})
    return c


def test_ensure_hist_ignore_l_entree_lot_et_rollback_restaure_les_tenseurs():
    c = _couche()
    hist = M.DecoderLayerGDN.ensure_hist(c, 3)
    assert set(hist) == {"conv", "S"} and hist["conv"].shape == (3, 4, 3)
    hist["conv"][1].fill_(7.0); hist["S"][1].fill_(9.0)
    M.DecoderLayerGDN.rollback(c, 2)
    assert float(c.static["conv"][0, 0]) == 7.0 and float(c.static["S"][0, 0, 0]) == 9.0 and c.static["lot"] == (0, 1)


def test_le_temoin_sans_filtre_cassait():
    c = _couche()
    try:
        {k: torch.zeros((2,) + tuple(v.shape)) for k, v in c.static.items() if k not in ("cache", "scores")}
    except AttributeError as e:
        assert "'tuple' object has no attribute 'shape'" in str(e)
    else:
        raise AssertionError("le témoin devait casser sur le tuple")
