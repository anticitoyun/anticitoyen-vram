"""Levier 1 (revue/poste1-levier-1-conception-21-09 § 4 et § 7) : le glouton
de `_sample_lent` capturé dans le graphe, DÉFAUT depuis d145bf0d ; témoin `ACVRAM_SAMPLER_LENT=1`.
Tests à sec (tenseurs CPU, faux graphes) ; ids/logprobs au bit sur carte et
frontière A/B : poste2. Ce qui doit casser : un noyau qui diverge de
`_sample_lent` (ids ou bits des logprobs), un tampon lu au-delà du lot réel
(godet ≠ tampon), un lot mêlé qui prend le paquet, un clone qui redevient une
vue (course clone/rejeu), deux rapatriements au lieu d un, une ligne de
régime qui dit `graphe` sans opt-in ou sans graphes, un défaut qui change."""
import os
from types import SimpleNamespace

import pytest
import torch

from acvram.engine.graphs import GraphRunner, depaqueter_logprobs, echantillon_glouton_dans
from acvram.engine.sampler import SamplingParams, _sample_lent


def _logits(n, v=97, graine=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(graine)
    l = torch.randn(n, v, generator=g) * 3
    l[0, 3] = l[0].max() + 0.0   # égalité stricte sur la ligne 0 : premier indice dans les deux chemins
    return l.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paquet_au_bit_avec_sample_lent(dtype):
    logits = _logits(12, dtype=dtype)
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    echantillon_glouton_dans(sortie, logits)
    ids_ref, lp_ref = _sample_lent(logits, [SamplingParams(temperature=0.0)] * 12)
    assert torch.equal(sortie[0, :12], ids_ref)
    assert depaqueter_logprobs(sortie[1, :12].tolist()) == lp_ref.tolist()   # bits fp32 exacts, pas ± eps
    assert int(sortie[0, 0]) == 3


def test_godet_different_du_tampon_fantomes_jamais_lus():
    """Lot réel 5 dans un godet 8 : 5 colonnes écrites, 3 intactes ; un godet 16
    après un godet 8 ne rend pas les ids du 8 (n vient du rejeu, pas du tampon)."""
    sortie8 = torch.full((2, 8), -7, dtype=torch.int64)
    echantillon_glouton_dans(sortie8, _logits(5))
    assert torch.equal(sortie8[:, 5:], torch.full((2, 3), -7, dtype=torch.int64))
    gr = GraphRunner.__new__(GraphRunner)
    gr.sampler_graphe = True
    gr._echantillon = (sortie8, 5)
    p = gr.prendre_echantillon()
    assert p.shape == (2, 5)
    sortie16 = torch.zeros(2, 16, dtype=torch.int64)
    echantillon_glouton_dans(sortie16, _logits(16, graine=1))
    gr._echantillon = (sortie16, 16)
    q = gr.prendre_echantillon()
    assert q.shape == (2, 16) and not torch.equal(q[0, :5], p[0])


def test_course_clone_rejeu_rend_faux_sans_clone():
    """Le paquet survit au rejeu suivant parce qu il est un CLONE : un rejeu
    qui réécrit le tampon ne le touche pas. Une vue (sans clone) rendrait les
    ids du pas suivant — c est ce que le témoin `vue` montre."""
    sortie = torch.zeros(2, 12, dtype=torch.int64)
    echantillon_glouton_dans(sortie, _logits(12, graine=2))
    gr = GraphRunner.__new__(GraphRunner)
    gr.sampler_graphe = True
    gr._echantillon = (sortie, 12)
    paquet = gr.prendre_echantillon()
    vue = sortie[:, :12]
    avant = paquet.clone()
    echantillon_glouton_dans(sortie, _logits(12, graine=3))        # « rejeu n+1 »
    assert torch.equal(paquet, avant), "le clone a bougé : course clone/rejeu"
    assert paquet._base is None
    assert not torch.equal(vue, avant), "le témoin sans clone devrait rendre les ids du pas suivant"
    assert gr.prendre_echantillon() is None, "une deuxième prise rend un paquet périmé"


def _seqs(params_list, finis=()):
    from acvram.engine.runner import Sequence
    seqs = [Sequence(prompt_ids=[1, 2, 3], params=p) for p in params_list]
    for i in finis:
        seqs[i].finished = True
    return seqs


class _FauxGraphes:
    def __init__(self, sortie, n, actif=True):
        self.sampler_graphe = actif
        self._echantillon = (sortie, n)
        self.prises = 0

    def prendre_echantillon(self):
        self.prises += 1
        return GraphRunner.prendre_echantillon(self)


def _moteur_nu():
    """Un Engine sans modèle : `_sample_only` et `_consommer` ne lisent que
    graphs, tokenizer, _eos, max_model_len."""
    from acvram.engine.runner import Engine
    eng = Engine.__new__(Engine)
    eng.tokenizer = None
    eng._eos = set()
    eng.max_model_len = 4096
    eng.pipeline_actif = True
    return eng


def test_lot_mele_prend_l_ancien_chemin_et_reste_au_bit():
    logits = _logits(12, graine=4)
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    echantillon_glouton_dans(sortie, logits)
    eng = _moteur_nu()
    eng.graphs = _FauxGraphes(sortie, 12)
    params = [SamplingParams(temperature=0.0)] * 11 + [SamplingParams(temperature=0.7)]
    seqs = _seqs(params)
    g = torch.Generator().manual_seed(0)
    tokens, lps = eng._sample_only(logits, seqs, depuis_graphe=True)
    assert eng.graphs.prises == 0, "un lot mêlé ne prend jamais le paquet"
    assert lps.dtype == torch.float32
    ref_ids, _ = _sample_lent(logits, params, None, generator=torch.Generator().manual_seed(0))
    assert torch.equal(tokens[:11], ref_ids[:11])                    # les gloutons du lot mêlé, au bit
    # une séquence finie (fantôme) dans le lot : ancien chemin aussi
    eng.graphs = _FauxGraphes(sortie, 12)
    eng._sample_only(logits, _seqs([SamplingParams(temperature=0.0)] * 12, finis=(3,)), depuis_graphe=True)
    assert eng.graphs.prises == 0
    # hors pipeline (depuis_graphe absent) : jamais le paquet
    eng._sample_only(logits, _seqs([SamplingParams(temperature=0.0)] * 12))
    assert eng.graphs.prises == 0


def test_lot_glouton_prend_le_paquet_un_seul_rapatriement_ids_puis_logprobs(monkeypatch):
    logits = _logits(12, graine=5)
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    echantillon_glouton_dans(sortie, logits)
    eng = _moteur_nu()
    eng.graphs = _FauxGraphes(sortie, 12)
    seqs = _seqs([SamplingParams(temperature=0.0, max_tokens=64)] * 12)
    tokens, lps = eng._sample_only(logits, seqs, depuis_graphe=True)
    assert eng.graphs.prises == 1 and lps.dtype == torch.int64 and lps._base is tokens._base
    rapatriements = []
    orig = torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "tolist", lambda t: (rapatriements.append(tuple(t.shape)), orig(t))[1])
    outs = eng._consommer(tokens, lps, seqs)
    assert rapatriements == [(2, 12)], f"un seul rapatriement attendu, vu {rapatriements}"
    ref_ids, ref_lp = _sample_lent(logits, [s.params for s in seqs])
    assert [o.token_ids[0] for o in outs] == ref_ids.tolist()
    assert [s.cumulative_logprob for s in seqs] == ref_lp.tolist()          # les logprobs, au bit, avec les ids


def test_defaut_graphe_et_temoin_lent(monkeypatch, capsys):
    """Défaut = graphe (verdict poste4 d145bf0d) ; ACVRAM_SAMPLER_LENT=1 =
    témoin ; l ancien nom ACVRAM_SAMPLER_GRAPHE est encore lu, avec un
    avertissement une fois (=1 : défaut ; =0 : témoin)."""
    from acvram.engine import graphs as G
    monkeypatch.delenv("ACVRAM_SAMPLER_GRAPHE", raising=False)
    monkeypatch.delenv("ACVRAM_SAMPLER_LENT", raising=False)
    assert G.sampler_graphe_actif() is True
    monkeypatch.setenv("ACVRAM_SAMPLER_LENT", "1")
    assert G.sampler_graphe_actif() is False
    monkeypatch.delenv("ACVRAM_SAMPLER_LENT")
    monkeypatch.setattr(G, "_AVERTI_SAMPLER_GRAPHE", False)
    monkeypatch.setenv("ACVRAM_SAMPLER_GRAPHE", "0")
    assert G.sampler_graphe_actif() is False
    assert "ACVRAM_SAMPLER_LENT=1" in capsys.readouterr().out
    monkeypatch.setenv("ACVRAM_SAMPLER_GRAPHE", "1")
    assert G.sampler_graphe_actif() is True and capsys.readouterr().out == ""   # averti une fois
    # le témoin : aucune prise, ancien chemin au bit
    eng = _moteur_nu()
    eng.graphs = _FauxGraphes(torch.zeros(2, 4, dtype=torch.int64), 4, actif=False)
    logits = _logits(4, graine=6)
    tokens, lps = eng._sample_only(logits, _seqs([SamplingParams(temperature=0.0)] * 4), depuis_graphe=True)
    assert eng.graphs.prises == 0 and lps.dtype == torch.float32
    assert torch.equal(tokens, _sample_lent(logits, [SamplingParams(temperature=0.0)] * 4)[0])
    assert G.depaqueter_logprobs([0]) == [0.0]


def test_ligne_de_regime_graphe_seulement_pipeline_et_graphes_et_opt_in(converted):
    from test_engine import _engine_cpu
    eng = _engine_cpu(converted)
    assert " sampler=lent " in eng.regime_ligne() + " "                     # sans graphes : lent, même par défaut
    eng.graphs = SimpleNamespace(sampler_graphe=True, enabled=True, raison="")
    eng.pipeline_actif = True
    assert " sampler=graphe " in eng.regime_ligne() + " "
    eng.pipeline_actif = False
    assert " sampler=lent " in eng.regime_ligne() + " "
    eng.pipeline_actif = True
    eng.graphs = SimpleNamespace(sampler_graphe=False, enabled=True, raison="")
    assert " sampler=lent " in eng.regime_ligne() + " "
