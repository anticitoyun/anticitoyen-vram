"""`MoEBlock._try_build_stacks` disait « formats mélangés » pour un bf16
UNIFORME (Jérôme, 15/09, relayant Manon : GLM-4.7-Flash-srcbf16-nvfp4
converti --format bf16, piles_ok=False, message trompeur — aucun mélange
n'existe, bf16 est simplement un format que ce chemin ne sait pas
empiler)."""

import torch

from acvram.engine.layers import QuantLinear
from acvram.engine.model import MLP, MoEBlock
from acvram.quant.formats import PlainTensor


def _plain(out_features, in_features):
    t = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
    return QuantLinear(PlainTensor(t, tuple(t.shape), "bf16"),
                      out_features=out_features, in_features=in_features)


def _moe_bf16_uniforme(n_experts=4):
    router = _plain(n_experts, 16)
    experts = [MLP(_plain(32, 16), _plain(32, 16), _plain(16, 32))
              for _ in range(n_experts)]
    return MoEBlock(router, experts, top_k=2)


def test_experts_bf16_uniformes_ne_sont_pas_dits_melanges():
    bloc = _moe_bf16_uniforme()
    ok = bloc._try_build_stacks()
    assert ok is False                       # attendu : pas de pile pour bf16
    assert "mélangés" not in bloc._raison_repli, bloc._raison_repli
    assert "uniforme" in bloc._raison_repli, bloc._raison_repli


def test_experts_vraiment_melanges_restent_signales_comme_tels():
    from acvram.quant.nvfp4 import NVFP4Tensor

    bloc = _moe_bf16_uniforme(n_experts=2)
    # Remplace UN seul expert par un format NVFP4 factice minimal : les
    # deux autres restent bf16 -> mélange réel, contrairement au test
    # ci-dessus où les 4 experts partagent le même format.
    fausse = object.__new__(NVFP4Tensor)
    bloc.experts[0].gate_proj.qweight = fausse
    ok = bloc._try_build_stacks()
    assert ok is False
    assert "mélangés" in bloc._raison_repli, bloc._raison_repli
