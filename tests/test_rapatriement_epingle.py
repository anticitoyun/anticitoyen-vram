"""Levier 2 (revue/poste1-levier-2-conception-22-09 § 3) : rapatriement des
ids/logprobs par tampon hôte épinglé à double parité, DÉFAUT depuis le
verdict poste4 22/09 ; témoin `ACVRAM_RAPATRIEMENT_FLUX=1`. Tests à sec (tenseurs CPU, faux graphes,
faux événement) ; carte : poste2. Ce qui doit casser : ids ou logprobs qui
divergent de `_sample_lent` sur 100 pas, un tampon de parité réutilisé avant
lecture sans lever, une copie enfilée sur le flux pendant `_consommer`, un
événement enregistré avant la copie, une ligne de régime qui dit `epingle`
sans les trois conditions, un défaut qui change."""
from types import SimpleNamespace

import pytest
import torch

from acvram.engine.graphs import GraphRunner, echantillon_glouton_dans
from acvram.engine.sampler import SamplingParams, _sample_lent


class _FauxGraphes:
    def __init__(self, sortie, actif=True):
        self.sampler_graphe = actif
        self.sortie = sortie
        self._echantillon = None
        self.enabled, self.raison = True, ""

    def rejouer(self, logits):
        echantillon_glouton_dans(self.sortie, logits)
        self._echantillon = (self.sortie, logits.shape[0])

    def prendre_echantillon(self):
        return GraphRunner.prendre_echantillon(self)


def _moteur(sortie, epingle=True, graphe=True):
    from acvram.engine.runner import Engine
    eng = Engine.__new__(Engine)
    eng.tokenizer, eng._eos, eng.max_model_len = None, set(), 4096
    eng.pipeline_actif = True
    eng.graphs = _FauxGraphes(sortie, actif=graphe)
    eng.rapatriement_epingle = epingle
    eng._epingles, eng._parite_epingle = [None, None], 0
    return eng


def _seqs(n, temp=0.0):
    from acvram.engine.runner import Sequence
    return [Sequence(prompt_ids=[1, 2], params=SamplingParams(temperature=temp, max_tokens=10_000)) for _ in range(n)]


class _Evenement:
    journal: list = []

    def record(self):
        _Evenement.journal.append("event")

    def synchronize(self):
        pass


@pytest.fixture(autouse=True)
def _faux_evenement(monkeypatch):
    _Evenement.journal = []
    monkeypatch.setattr(torch.cuda, "Event", _Evenement)


def _pas(eng, logits, seqs):
    """Un pas du pipeline, réduit à ce que le levier touche : rejeu (le tampon
    statique est RÉÉCRIT), prise, copie épinglée + événement, consommation."""
    eng.graphs.rejouer(logits)
    tokens, lps = eng._sample_only(logits, seqs, depuis_graphe=True)
    epingle, ev = eng._apres_echantillon(tokens, lps)
    return tokens, lps, epingle, ev


def test_ids_et_logprobs_au_bit_sur_cent_pas():
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(12)
    for pas in range(100):
        logits = torch.randn(12, 131, generator=torch.Generator().manual_seed(pas)) * 4
        ref_ids, ref_lp = _sample_lent(logits, [s.params for s in seqs])
        tokens, lps, epingle, ev = _pas(eng, logits, seqs)
        assert epingle is not None and epingle["n"] == 12
        # « rejeu n+1 » avant la consommation de n : le tampon statique change, l épinglé de n non
        echantillon_glouton_dans(sortie, torch.randn(12, 131) * 4)
        ev.synchronize()
        outs = eng._consommer(tokens, lps, seqs, epingle=epingle)
        assert [o.token_ids[0] for o in outs] == ref_ids.tolist(), pas
        assert [s.output_ids[-1] for s in seqs] == ref_ids.tolist()
        assert [round(s.cumulative_logprob - s.cumulative_logprob + float(l), 6) for s, l in zip(seqs, ref_lp.tolist())] \
            == [round(float(l), 6) for l in ref_lp.tolist()]
        for s in seqs:
            s.cumulative_logprob = 0.0
    assert all(len(s.output_ids) == 100 for s in seqs)


def test_logprobs_cumules_au_bit():
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(4)
    logits = torch.randn(4, 50, generator=torch.Generator().manual_seed(9))
    _, ref_lp = _sample_lent(logits, [s.params for s in seqs])
    tokens, lps, epingle, ev = _pas(eng, logits, seqs)
    eng._consommer(tokens, lps, seqs, epingle=epingle)
    assert [s.cumulative_logprob for s in seqs] == ref_lp.tolist()


def test_parite_double_tampon_et_garde_qui_leve():
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(3)
    l0 = torch.randn(3, 40, generator=torch.Generator().manual_seed(0))
    l1 = torch.randn(3, 40, generator=torch.Generator().manual_seed(1))
    t0, p0, e0, _ = _pas(eng, l0, seqs)            # pas n → parité 0
    t1, p1, e1, _ = _pas(eng, l1, seqs)            # pas n+1 → parité 1, enfilé AVANT la lecture de n
    assert e0 is not e1 and e0 is eng._epingles[0] and e1 is eng._epingles[1]
    ids0 = _sample_lent(l0, [s.params for s in seqs])[0].tolist()
    assert e0["tenseur"][0, :3].tolist() == ids0, "l épinglé de n a été écrasé par n+1"
    eng._consommer(t0, p0, seqs, epingle=e0)       # lecture de n
    assert e0["lu"] is True and e1["lu"] is False
    # cibler la parité 0 : n est lu → autorisé ; puis cibler la parité 1 SANS avoir lu n+1 → lève
    _pas(eng, l0, seqs)                              # parité 0 à nouveau : ok
    with pytest.raises(RuntimeError, match="réutilisé avant lecture"):
        _pas(eng, l1, seqs)                          # parité 1, e1 jamais lu


def test_tampon_epingle_exactement_2_n_et_contigu():
    """poste2 22/09 (levier 2 réfuté à la frontière : suite_prep 151 → 6 860 µs) :
    la vue `[:, :n]` d un tampon [2, 16] n est pas contiguë, la copie D2H
    passait par un tampon paginable et devenait synchrone. Le tampon fait
    exactement [2, n], contigu, un par (parité, n) ; une cible d une autre
    forme est refusée, jamais copiée en silence."""
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(12)
    _, _, e, _ = _pas(eng, torch.randn(12, 30), seqs)
    assert tuple(e["tenseur"].shape) == (2, 12) and e["tenseur"].is_contiguous()
    # recomposition : n passe à 5 → nouveau tampon [2, 5] sur la parité suivante
    e["lu"] = True
    for s in eng._epingles:
        if s is not None:
            s["lu"] = True
    _, _, e5, _ = _pas(eng, torch.randn(5, 30), _seqs(5))
    assert tuple(e5["tenseur"].shape) == (2, 5) and e5["tenseur"].is_contiguous()
    # garde : une cible d une autre forme lève au lieu de copier une vue non contiguë
    eng2 = _moteur(sortie)
    eng2._epingles = [{"tenseur": torch.zeros(2, 16, dtype=torch.int64)[:, :12], "lu": True, "n": 12}, None]
    eng2.graphs.rejouer(torch.randn(12, 30))
    tokens, lps = eng2._sample_only(torch.randn(12, 30), _seqs(12), depuis_graphe=True)
    with pytest.raises(RuntimeError, match="contiguës"):
        eng2._apres_echantillon(tokens, lps)


def test_consommer_n_enfile_rien_sur_le_flux(monkeypatch):
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(5)
    tokens, lps, epingle, ev = _pas(eng, torch.randn(5, 30), seqs)
    appels = []
    orig_tolist, orig_cpu = torch.Tensor.tolist, torch.Tensor.cpu
    monkeypatch.setattr(torch.Tensor, "tolist", lambda t: (appels.append(("tolist", t is epingle["tenseur"] or t._base is epingle["tenseur"])), orig_tolist(t))[1])
    monkeypatch.setattr(torch.Tensor, "cpu", lambda t, *a, **k: (appels.append(("cpu", False)), orig_cpu(t, *a, **k))[1])
    eng._consommer(tokens, lps, seqs, epingle=epingle)
    assert appels == [("tolist", True)], appels     # une seule lecture, sur le tampon épinglé, aucun .cpu()


def test_evenement_enregistre_apres_la_copie(monkeypatch):
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    seqs = _seqs(2)
    orig = torch.Tensor.copy_

    def copie(t, src, non_blocking=False):
        _Evenement.journal.append(("copy_", non_blocking))
        return orig(t, src, non_blocking=non_blocking)
    monkeypatch.setattr(torch.Tensor, "copy_", copie)
    eng.graphs.rejouer(torch.randn(2, 20))
    tokens, lps = eng._sample_only(torch.randn(2, 20), seqs, depuis_graphe=True)
    _Evenement.journal.clear()
    eng._apres_echantillon(tokens, lps)
    assert _Evenement.journal == [("copy_", True), "event"], _Evenement.journal


def test_sans_opt_in_ou_lot_ineligible_rien_ne_change():
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie, epingle=False)
    seqs = _seqs(4)
    tokens, lps, epingle, ev = _pas(eng, torch.randn(4, 30), seqs)
    assert epingle is None and _Evenement.journal == ["event"] and eng._epingles == [None, None]
    eng = _moteur(sortie, epingle=True)
    tokens, lps, epingle, ev = _pas(eng, torch.randn(4, 30), _seqs(4, temp=0.8))   # lot inéligible : logprobs fp32
    assert epingle is None and lps.dtype == torch.float32 and eng._parite_epingle == 0


def test_defaut_epingle_et_temoin_flux(monkeypatch, capsys):
    """Défaut = épinglé (verdict poste4 22/09) ; ACVRAM_RAPATRIEMENT_FLUX=1 =
    témoin ; l ancien ACVRAM_RAPATRIEMENT_EPINGLE encore lu, averti une fois."""
    from acvram.engine import graphs as G
    monkeypatch.delenv("ACVRAM_RAPATRIEMENT_EPINGLE", raising=False)
    monkeypatch.delenv("ACVRAM_RAPATRIEMENT_FLUX", raising=False)
    assert G.rapatriement_epingle_actif() is True
    monkeypatch.setenv("ACVRAM_RAPATRIEMENT_FLUX", "1")
    assert G.rapatriement_epingle_actif() is False
    monkeypatch.delenv("ACVRAM_RAPATRIEMENT_FLUX")
    monkeypatch.setattr(G, "_AVERTI_RAPATRIEMENT", False)
    monkeypatch.setenv("ACVRAM_RAPATRIEMENT_EPINGLE", "0")
    assert G.rapatriement_epingle_actif() is False and "RAPATRIEMENT_FLUX=1" in capsys.readouterr().out
    monkeypatch.setenv("ACVRAM_RAPATRIEMENT_EPINGLE", "1")
    assert G.rapatriement_epingle_actif() is True and capsys.readouterr().out == ""


def test_ligne_de_regime_epingle_sous_trois_conditions(converted, monkeypatch):
    from test_engine import _engine_cpu
    monkeypatch.delenv("ACVRAM_RAPATRIEMENT_EPINGLE", raising=False)
    monkeypatch.delenv("ACVRAM_RAPATRIEMENT_FLUX", raising=False)
    eng = _engine_cpu(converted)
    # défaut = épinglé, mais la ligne ne le dit que sous pipeline ET graphes ET sampler=graphe : sans graphes → flux
    assert eng.rapatriement_epingle is True and " rapatriement=flux " in eng.regime_ligne() + " "
    eng.graphs = SimpleNamespace(sampler_graphe=True, enabled=True, raison="")
    eng.pipeline_actif, eng.rapatriement_epingle = True, True
    assert " rapatriement=epingle " in eng.regime_ligne() + " "
    eng.graphs.sampler_graphe = False
    assert " rapatriement=flux " in eng.regime_ligne() + " "
    eng.graphs.sampler_graphe, eng.pipeline_actif = True, False
    assert " rapatriement=flux " in eng.regime_ligne() + " "
