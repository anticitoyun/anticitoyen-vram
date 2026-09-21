"""Porte FP8-MLA (sage-cloture-23h59-19-09) : `ACVRAM_MLA_A8=e4m3|int8` arrondit
l'ENTRÉE de q_b, kv_a et o (engine/mla.py `_a8`), les modules gardent leur nom
dans l'arbre ; identité au défaut ; la table de régime porte la variable."""
import os
import re
import sys
import pathlib

import torch

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from acvram.engine import mla as MLA                                            # noqa: E402
from acvram.quant.fakequant_activation import fake_quantize_a8, fake_quantize_e4m3_activation   # noqa: E402


def test_identite_au_defaut_et_arrondi_sous_la_porte(monkeypatch):
    x = (torch.randn(6, 3, 512) * 0.7).to(torch.bfloat16)
    monkeypatch.setattr(MLA, "_MLA_A8", "off")
    assert MLA._a8(x) is x
    monkeypatch.setattr(MLA, "_MLA_A8", "e4m3")
    y = MLA._a8(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.equal(y, fake_quantize_e4m3_activation(x.reshape(-1, 512)).reshape(x.shape))
    assert not torch.equal(y, x)                                     # la porte change l'entrée
    monkeypatch.setattr(MLA, "_MLA_A8", "int8")
    assert torch.equal(MLA._a8(x), fake_quantize_a8(x.reshape(-1, 512), "int8").reshape(x.shape))


def test_les_trois_projections_passent_par_la_porte_et_gardent_leur_nom():
    src = (RACINE / "acvram" / "engine" / "mla.py").read_text()
    # aucun appel direct ne contourne la porte ; la registration des modules reste nominale
    # le seul appel direct est celui de la porte elle-même (`_a8(` juste derrière)
    assert re.findall(r"self\.o_proj\((?!_a8\()", src) == [] and len(re.findall(r"self\._o\(", src)) >= 7
    assert re.findall(r"self\.kv_a_proj\((?!_a8\()", src) == [] and "self._kv_a(" in src
    assert re.findall(r"self\.q_b_proj\((?!_a8\()", src) == [] and "self._q_b(" in src
    assert "self.q_proj, self.kv_a_proj, self.o_proj = q_proj, kv_a_proj, o_proj" in src


def test_variable_dans_la_table_de_regime():
    from acvram import regime
    v = {x.env: x for x in regime.VARIABLES}["ACVRAM_MLA_A8"]
    assert v.defaut == "off" and v.lu_a == ("acvram.engine.mla", "_MLA_A8")
    assert "ACVRAM_MLA_A8" in regime.regime_noyaux()["variables"]
