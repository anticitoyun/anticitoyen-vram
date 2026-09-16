"""Juge de la fusion (1) du poste F (kernels/route_prep.py) : `eid`, le
compteur d'usage des experts et les index de jetons doivent être ceux du
chemin torch (`masked_fill(~valid, -1)`, `_compter_routage`, `arange`,
`repeat_interleave`) AU BIT, sur les godets 1/2/8/16 avec fantômes, avec et
sans `valid`, compteur cumulé sur plusieurs pas ; et un bras qui doit
casser : un fantôme non masqué change le compteur. Sans carte :
``TRITON_INTERPRET=1`` (conftest)."""
import importlib
import os

import pytest
import torch


def _rp():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    rp = importlib.import_module("acvram.kernels.route_prep")
    if not rp.disponible():
        pytest.skip("Triton indisponible")
    return rp


def _torch(topi, valid, usage):
    """Le chemin torch d'aujourd'hui (model.py : forward + _compter_routage + _forward_grouped)."""
    if valid is not None:
        topi = topi.masked_fill(~valid.unsqueeze(-1), -1)
    idx = topi.reshape(-1).to(torch.int64)
    usage.scatter_add_(0, idx.clamp(min=0), (idx >= 0).to(torch.int64))
    t, k = topi.shape
    eid = topi.reshape(-1).to(torch.int32)
    tok = torch.arange(t, dtype=torch.int32).repeat_interleave(k)
    seq = torch.arange(t * k, dtype=torch.int32)
    return eid, tok, seq


@pytest.mark.parametrize("godet", [1, 2, 8, 16])
@pytest.mark.parametrize("avec_valid", [True, False])
def test_eid_compteur_et_index_au_bit(godet, avec_valid):
    rp = _rp()
    torch.manual_seed(godet)
    E, k = 128, 8
    usage_t = torch.zeros(E, dtype=torch.int64)
    usage_r = torch.zeros(E, dtype=torch.int64)
    for pas in range(3):                                   # compteur cumulé sur trois pas
        topi = torch.randint(0, E, (godet, k), dtype=torch.int32)
        valid = None
        if avec_valid:
            valid = torch.ones(godet, dtype=torch.bool)
            valid[max(1, godet - godet // 4):] = False      # les dernières lignes sont des fantômes
        eid_t, tok_t, seq_t = _torch(topi.clone(), valid, usage_t)
        eid_r = rp.route_prep(topi, valid, usage_r)
        tok_r, seq_r = rp.index_jetons(godet, k, topi.device)
        assert torch.equal(eid_r, eid_t) and eid_r.dtype == torch.int32
        assert torch.equal(tok_r, tok_t) and torch.equal(seq_r, seq_t)
        assert torch.equal(usage_r, usage_t), (usage_r - usage_t).nonzero()
    assert rp.index_jetons(godet, k, topi.device)[0] is tok_r, "les index sont réservés une fois par godet"


def test_un_fantome_non_masque_casse_le_compteur():
    """Le bras qui doit différer : sans `valid`, les fantômes comptent — le
    juge voit la différence, donc il voit aussi un masque oublié."""
    rp = _rp()
    topi = torch.randint(0, 16, (8, 4), dtype=torch.int32)
    valid = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    avec, sans = torch.zeros(16, dtype=torch.int64), torch.zeros(16, dtype=torch.int64)
    e1 = rp.route_prep(topi, valid, avec)
    e2 = rp.route_prep(topi, None, sans)
    assert int(avec.sum()) == 24 and int(sans.sum()) == 32
    assert (e1 == -1).sum() == 8 and (e2 == -1).sum() == 0
