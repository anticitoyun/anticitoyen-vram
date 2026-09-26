"""22/09 : un converti MoE dont les experts portent des échelles AWQ DISTINCTES
entre `gate_proj` et `up_proj` perd la disposition Marlin et retombe sur la
pile naturelle (Qwen3-Coder-30B-A3B-nvfp4-qkv-22-09 : 8,62 ms/pas contre 6,7).

Ce test fixe la condition exacte, à sec, sur un bloc jouet :
* échelles gate == up (ce que pose la pré-passe d alpha commun hors experts) :
  `up_distinct` faux, et le refus Marlin ne vient QUE de l absence de carte ;
* une seule échelle d up changée sur UN expert sur quatre : `up_distinct`
  devient vrai (la table est [E, K], `torch.equal` est global) et le refus
  nomme « gate/up à entrées distinctes ».
Il casse si l on rend la fusion tolérante sans traiter le GEMV fusionné : ce
serait une sortie fausse au décodage, pas un gain.
"""
from __future__ import annotations

import torch

from acvram.engine.moe import MoEBlock


def _bloc(awq_distinct: bool):
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    from acvram.quant.calibrate import ChannelScaler
    bloc = _bloc_moe_jouet(E=4, H=128, I=64, top_k=2, dev="cpu", awq=True)
    if awq_distinct:
        # un seul expert, une seule projection : l échelle d up cesse d être
        # celle de gate (le convertisseur cherche l alpha par projection sur
        # les experts — quant/convert.py:543-544 les exclut de l alpha commun)
        e = bloc.experts[1]
        s = e.up_proj.scaler.scale.clone()
        s[0] = s[0] * 1.5
        e.up_proj.scaler = ChannelScaler(s, 0)
    return bloc


def _construire(bloc, capsys):
    MoEBlock._marlin_refus_dit = False
    assert bloc._try_build_stacks()
    return capsys.readouterr().out


def test_gate_up_partagees_gardent_marlin(capsys):
    bloc = _bloc(awq_distinct=False)
    sortie = _construire(bloc, capsys)
    assert bloc._stacks_awq["up_distinct"] is False
    assert bloc._stacks_marlin is None                      # à sec : pas de carte
    assert "pile hors CUDA" in sortie, sortie               # et c est la SEULE raison
    assert "entrées distinctes" not in sortie


def test_une_seule_echelle_distincte_fait_tomber_marlin(capsys):
    bloc = _bloc(awq_distinct=True)
    sortie = _construire(bloc, capsys)
    assert bloc._stacks_awq["up_distinct"] is True
    assert bloc._stacks_marlin is None
    assert "entrées distinctes" in sortie, sortie
    assert getattr(bloc, "experts_layout", "naturel") == "naturel"


def test_table_unite_ne_compte_pas_comme_distincte(capsys):
    """Un converti dont TOUTES les échelles valent 1 (repli identité partout,
    Qwen3-Coder-nvfp4-qkvo-i8c du 18/09 : 18 432 experts sans stats) garde
    Marlin : la table d unité est écartée (`awq[nom] = None`), pas comparée."""
    from acvram.quant.calibrate import ChannelScaler
    bloc = _bloc(awq_distinct=False)
    for e in bloc.experts:
        for nom in ("gate_proj", "up_proj", "down_proj"):
            p = getattr(e, nom)
            p.scaler = ChannelScaler(torch.ones_like(p.scaler.scale), 0)
    sortie = _construire(bloc, capsys)
    assert bloc._stacks_awq["gate_proj"] is None and bloc._stacks_awq["up_distinct"] is False
    assert "entrées distinctes" not in sortie
