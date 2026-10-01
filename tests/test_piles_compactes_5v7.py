"""5v7 (30/09) : les petits tenseurs qui survivent aux piles d experts (échelles AWQ par expert, global_scale) épinglaient
les segments de 2 Mio du bassin des petits blocs : 4,58 Gio « nets » sur Coder-30B, réservés non alloués. Ils sont
regroupés en un tampon par (appareil, dtype), par vues, avant que le cache soit rendu. À sec : regroupement, valeurs,
témoin, appel depuis `_try_build_stacks` ; le gain mémoire se prend sur carte (scellé revue/poste1-5v7-scelle-30-09.md)."""
from __future__ import annotations

import torch

from acvram.engine import moe as MOE
from acvram.engine.layers import ChannelScaler, QuantLinear
from acvram.engine.model import MLP, MoEBlock
from acvram.quant.nvfp4 import quantize_nvfp4


def _bloc(E=6, H=128, I=64):
    def lin(o, i, g):
        gen = torch.Generator().manual_seed(g)
        w = (torch.randn(o, i, generator=gen) * 0.02).to(torch.bfloat16)
        sc = ChannelScaler((0.5 + torch.rand(i, generator=gen)).to(torch.bfloat16), 0)
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i, scaler=sc).to_device("cpu")
    experts = [MLP(lin(I, H, 10 * e + 1), lin(I, H, 10 * e + 2), lin(H, I, 10 * e + 3)) for e in range(E)]
    return MoEBlock(lin(E, H, 999), experts, top_k=2)


def _survivants(bloc):
    out = []
    for e in bloc.experts:
        for nom in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(e, nom)
            out += [lin.scaler.scale, lin.qweight.global_scale]
    return out


def test_survivants_regroupes_par_vues_memes_valeurs():
    # Casse sur main : `_compacter_survivants` n existe pas (chaque survivant garde sa propre allocation)
    bloc = _bloc()
    avant = [x.clone() for x in _survivants(bloc)]
    n = bloc._compacter_survivants(sur_cpu=True)
    apres = _survivants(bloc)
    assert n == len(apres)
    assert all(torch.equal(a, b) and a.dtype == b.dtype and a.shape == b.shape for a, b in zip(avant, apres))
    assert len({x.untyped_storage().data_ptr() for x in apres[0::2]}) == 1, "échelles AWQ : un seul tampon"
    assert len({x.untyped_storage().data_ptr() for x in apres[1::2]}) == 1, "global_scale : un seul tampon"


def test_temoin_ne_touche_a_rien(monkeypatch):
    monkeypatch.setenv("ACVRAM_PILES_COMPACTER", "0")
    bloc = _bloc()
    ptrs = [x.untyped_storage().data_ptr() for x in _survivants(bloc)]
    assert bloc._compacter_survivants(sur_cpu=True) == 0
    assert [x.untyped_storage().data_ptr() for x in _survivants(bloc)] == ptrs


def test_hors_carte_rien_par_defaut():
    """Sur le processeur, pas d allocateur à segments : rien n est déplacé (appel réel de `_try_build_stacks`)."""
    bloc = _bloc()
    ptrs = [x.untyped_storage().data_ptr() for x in _survivants(bloc)]
    assert bloc._compacter_survivants() == 0
    assert [x.untyped_storage().data_ptr() for x in _survivants(bloc)] == ptrs


def test_appele_par_la_construction_des_piles(monkeypatch):
    appels = []
    monkeypatch.setattr(MOE.MoEBlock, "_compacter_survivants", lambda self, sur_cpu=False: appels.append(1) or 0)
    bloc = _bloc()
    assert bloc._try_build_stacks() is True
    assert appels == [1]


def test_le_cache_d_echelle_ne_garde_pas_l_ancienne_adresse():
    bloc = _bloc()
    sc = bloc.experts[0].gate_proj.scaler
    sc._au_dtype(torch.float32)                          # une conversion en cache, comme après un forward
    bloc._compacter_survivants(sur_cpu=True)
    assert "_cache_dtype" not in sc.__dict__
