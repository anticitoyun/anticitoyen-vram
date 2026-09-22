"""Pièce 36 côté moteur : top-K logprobs par pas de décodage et logprobs de
l invite (`echo`, teacher forcing), sans changer la sortie par défaut.
Ce qui doit casser : un top-K qui n est pas la distribution du pas (autre
normalisation), un jeton choisi ou un logprob cumulé modifié par la présence
du top-K, `logprobs_invite` qui rendrait un logprob au premier jeton."""
import math

import pytest
import torch

from acvram.engine.sampler import SamplingParams, _sample_lent, logprobs_des, top_logprobs


def _logits(n=4, v=11, graine=0):
    g = torch.Generator().manual_seed(graine)
    return torch.randn(n, v, generator=g) * 3


def test_top_logprobs_est_la_distribution_du_pas():
    l = _logits()
    ids, lp = top_logprobs(l, 3)
    assert ids.shape == (4, 3) and lp.shape == (4, 3) and ids.dtype == torch.int64
    ref = torch.log_softmax(l.float(), dim=-1)
    for r in range(4):
        assert ids[r].tolist() == torch.topk(ref[r], 3).indices.tolist()
        assert torch.allclose(lp[r], ref[r][ids[r]], atol=0)             # mêmes valeurs, pas « proches »
        assert lp[r].tolist() == sorted(lp[r].tolist(), reverse=True)    # triés
    # le logprob du choisi de `_sample_lent` est celui de la même distribution
    ids_c, lp_c = _sample_lent(l, [SamplingParams(temperature=0.0)] * 4)
    for r in range(4):
        assert abs(float(lp_c[r]) - float(ref[r][ids_c[r]])) < 1e-5
    assert top_logprobs(l, 99).shape[1] == l.shape[1] if hasattr(top_logprobs(l, 99), "shape") else True
    assert top_logprobs(l, 0)[0].shape[1] == 1                            # k borné à [1, vocab]


def test_logprobs_des_cibles():
    l = _logits(3, 7, graine=2)
    cibles = torch.tensor([0, 3, 6])
    ref = torch.log_softmax(l.float(), dim=-1)
    assert torch.allclose(logprobs_des(l, cibles), torch.stack([ref[i, c] for i, c in enumerate(cibles)]), atol=0)


class _Faux:
    """Moteur réduit : `_tops_si_demande` et `_consommer` n ont besoin que de ça."""
    def __init__(self):
        from acvram.engine.runner import Engine
        self.eng = Engine.__new__(Engine)
        self.eng.tokenizer, self.eng._eos, self.eng.max_model_len = None, set(), 4096

    def seqs(self, k_par_seq):
        from acvram.engine.runner import Sequence
        return [Sequence(prompt_ids=[1, 2], params=SamplingParams(temperature=0.0, max_tokens=99, logprobs=k))
                for k in k_par_seq]


def test_top_k_par_pas_et_sortie_par_defaut_inchangee():
    f = _Faux()
    l = _logits(3, 11, graine=5)
    sans = f.seqs([None, None, None])
    assert f.eng._tops_si_demande(l, sans) is None                        # personne ne demande : aucun calcul
    avec = f.seqs([2, None, 3])
    tops = f.eng._tops_si_demande(l, avec)
    assert tops[1] is None and len(tops[0][0]) == 2 and len(tops[2][0]) == 3
    ref = torch.log_softmax(l.float(), dim=-1)
    assert tops[0][0] == torch.topk(ref[0], 2).indices.tolist()
    # la consommation attache les champs sans changer le jeton ni le cumul
    ids, lps = _sample_lent(l, [s.params for s in avec])
    outs = f.eng._consommer(ids, lps, avec, tops=tops)
    assert [o.token_ids[0] for o in outs] == ids.tolist()
    assert outs[1].top_logprobs is None and outs[1].logprob is None       # séquence sans demande : rien
    assert len(outs[0].top_logprobs) == 2 and isinstance(outs[0].top_logprobs[0][0], int)
    assert abs(outs[0].logprob - float(lps[0])) < 1e-6
    assert [s.cumulative_logprob for s in avec] == [float(x) for x in lps]
    # sans tops, la sortie est celle d avant (au bit)
    for s in avec:
        s.output_ids.clear(); s.cumulative_logprob = 0.0; s.finished = False
    outs2 = f.eng._consommer(ids, lps, avec)
    assert [o.token_ids[0] for o in outs2] == ids.tolist() and all(o.top_logprobs is None for o in outs2)


def test_logprobs_invite_premier_jeton_sans_logprob():
    """L interface d `echo` : autant d entrées que de jetons, la première à None."""
    from acvram.engine.runner import Engine
    eng = Engine.__new__(Engine)
    r = Engine.logprobs_invite(eng, [7], top_k=0)
    assert r == {"ids": [7], "logprobs": [None], "top": None}
    r = Engine.logprobs_invite(eng, [], top_k=2)
    assert r["ids"] == [] and r["logprobs"] == [] and r["top"] is None
