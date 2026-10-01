"""Pièce mtp (poste3, 01/10, ordre chef après le bras long raté sur acvram-qwen3.8-27b-nvfp4) :
proposed_tokens=0 ne distingue pas « jamais essayé » de « essayé, rien à proposer ». Chaque
retour précoce du pas spéculatif pose maintenant une raison (Proposal.raison) ou un compteur
dédié (EngineStats.spec_hybride_hors_graphe, spec_k_insuffisant), exposés dans /metrics. À sec :
aucune carte, aucun modèle réel — un jouet par branche, qui casse si on retire le compteur."""
import types

import torch

from acvram.engine import runner as runner_mod
from acvram.engine.speculative import MTPProposer, NGramProposer, Proposal, _DraftState
from test_mla_glue_c15 import _jouet


def _seq(max_tokens=256, output_ids=None, ids=(1, 2, 3), seq_id=1):
    s = types.SimpleNamespace(id=seq_id, params=types.SimpleNamespace(max_tokens=max_tokens),
                               output_ids=list(output_ids or []))
    s.all_ids = list(ids)
    return s


# -- NGramProposer : raisons -------------------------------------------------

def test_ngram_raison_longueur():
    p = NGramProposer()
    prop = p.propose(_seq(ids=(1, 2)), k=4)   # trop court pour min_ngram+1
    assert prop.tokens == [] and prop.raison == "longueur"


def test_ngram_raison_veille():
    p = NGramProposer(adaptatif=True, pause=5)
    seq = _seq(ids=tuple(range(20)))
    st = p._st(seq)
    st["veille"] = 3
    prop = p.propose(seq, k=4)
    assert prop.tokens == [] and prop.raison == "veille"


def test_ngram_raison_aucun_match():
    p = NGramProposer()
    prop = p.propose(_seq(ids=(1, 2, 3, 4, 5, 6, 7, 8)), k=4)   # jamais vu, aucun suffixe indexé
    assert prop.tokens == [] and prop.raison == "aucun_match"


# -- MTPProposer : raisons ----------------------------------------------------

def _mtp(num_blocks=8, mtp_hidden=None, mtp_prefill=None):
    cache = types.SimpleNamespace(cfg=types.SimpleNamespace(num_blocks=num_blocks))
    tete = types.SimpleNamespace(cache=cache)
    model = types.SimpleNamespace(mtp=tete, _mtp_hidden=mtp_hidden, _mtp_hidden_n=0,
                                  _mtp_prefill=mtp_prefill, embed_tokens=None, lm_head=None)
    return MTPProposer(model, max_model_len=4096)


def test_mtp_raison_longueur():
    p = _mtp()
    p.max_model_len = 3
    prop = p.propose(_seq(ids=(1, 2, 3, 4, 5)), k=4)
    assert prop.tokens == [] and prop.raison == "longueur"


def test_mtp_raison_sans_hidden():
    p = _mtp(mtp_hidden=None)   # avant le premier pas : jamais rempli
    prop = p.propose(_seq(ids=(1, 2, 3)), k=4)
    assert prop.tokens == [] and prop.raison == "sans_hidden"


def test_mtp_raison_amorcage():
    h = torch.zeros(16, 8)   # hidden non vide : passe la garde sans_hidden
    p = _mtp(mtp_hidden=h, mtp_prefill=None)   # _mtp_prefill absent : _amorcer échoue
    prop = p.propose(_seq(ids=(1, 2, 3, 4, 5)), k=4)   # len(ids) > 2 : _amorcer est appelé
    assert prop.tokens == [] and prop.raison == "amorcage"


def test_mtp_raison_aucun_jeton():
    h = torch.zeros(16, 8)
    p = _mtp(num_blocks=0, mtp_hidden=h)   # allocator à sec : _ensure_blocks échoue dès le 1er jeton
    prop = p.propose(_seq(ids=(1, 2)), k=4)   # len(ids)<=2 : pas d'amorçage requis
    assert prop.tokens == [] and prop.raison == "aucun_jeton"


# -- runner.py : gardes du pas spéculatif (sans modèle, sans graphe réel) ----

def _engine(hyb, graphs, speculator, spec_k=4):
    eng = types.SimpleNamespace()
    eng.est_hybride = hyb
    eng.graphs = graphs
    eng.speculator = speculator
    eng.spec_k = spec_k
    eng.stats = runner_mod.EngineStats()
    eng._plain_decode = lambda decodable: "PLAIN"
    eng._speculative_decode = types.MethodType(runner_mod.Engine._speculative_decode, eng)
    return eng


def test_runner_hybride_hors_graphe_sans_graphes():
    eng = _engine(hyb=True, graphs=None, speculator=None)
    out = eng._speculative_decode([_seq()])
    assert out == "PLAIN" and eng.stats.spec_hybride_hors_graphe == 1


def test_runner_hybride_hors_graphe_lot_deux():
    graphs = types.SimpleNamespace(enabled=True, max_ql=None)
    eng = _engine(hyb=True, graphs=graphs, speculator=None)
    out = eng._speculative_decode([_seq(), _seq()])
    assert out == "PLAIN" and eng.stats.spec_hybride_hors_graphe == 1


def test_runner_raison_vide_propagee():
    graphs = types.SimpleNamespace(enabled=True, max_ql=None)
    speculator = types.SimpleNamespace(propose=lambda seq, k: Proposal([], raison="sans_hidden"))
    eng = _engine(hyb=True, graphs=graphs, speculator=speculator)
    out = eng._speculative_decode([_seq()])
    assert out == "PLAIN" and eng.stats.spec_raisons == {"sans_hidden": 1}


def test_runner_k_insuffisant():
    graphs = types.SimpleNamespace(enabled=True, max_ql=None)
    speculator = types.SimpleNamespace(propose=lambda seq, k: Proposal([7]))   # non vide
    eng = _engine(hyb=True, graphs=graphs, speculator=speculator, spec_k=4)
    # budget = max_tokens - len(output_ids) - 1 = 2 - 0 - 1 = 1 < spec_k=4
    out = eng._speculative_decode([_seq(max_tokens=2)])
    assert out == "PLAIN" and eng.stats.spec_k_insuffisant == 1


# -- compteurs permanents (01/10, après l'essai où "amorcage" domine) -------

def test_mtp_amorcage_echec_prefill_absent_releve_cause_et_tailles():
    h = torch.zeros(16, 8)
    p = _mtp(mtp_hidden=h, mtp_prefill=None)
    p.propose(_seq(ids=(1, 2, 3, 4, 5), seq_id=7), k=4)
    assert len(p._amorcage_echecs) == 1
    r = p._amorcage_echecs[-1]
    assert r == {"seq_id": 7, "cause": "prefill_absent", "hs_shape0": None,
                 "len_ids": 5, "len_ids_moins_1": 4}


def test_mtp_amorcage_echec_prefill_court_releve_tailles():
    h = torch.zeros(16, 8)
    prefill_court = torch.zeros(2, 8)   # trop court pour len(ids)-1 = 4
    p = _mtp(mtp_hidden=h, mtp_prefill=prefill_court)
    p.propose(_seq(ids=(1, 2, 3, 4, 5), seq_id=7), k=4)
    r = p._amorcage_echecs[-1]
    assert r == {"seq_id": 7, "cause": "prefill_court", "hs_shape0": 2,
                 "len_ids": 5, "len_ids_moins_1": 4}


def test_mtp_premier_echec_amorcage_garde_le_premier_pas_les_retentatives():
    h = torch.zeros(16, 8)
    p = _mtp(mtp_hidden=h, mtp_prefill=None)
    seq = _seq(ids=(1, 2, 3, 4, 5), seq_id=9)
    p.propose(seq, k=4)
    premier = dict(p._premier_echec_amorcage[9])
    seq.all_ids = [1, 2, 3, 4, 5, 6, 7]   # la séquence avance, une retentative de plus
    p.propose(seq, k=4)
    assert p._premier_echec_amorcage[9] == premier   # inchangé malgré la 2e tentative
    assert len(p._amorcage_echecs) == 2              # le relevé glissant, lui, avance


def test_mtp_premier_echec_amorcage_une_entree_par_sequence():
    h = torch.zeros(16, 8)
    p = _mtp(mtp_hidden=h, mtp_prefill=None)
    p.propose(_seq(ids=(1, 2, 3), seq_id=1), k=4)
    p.propose(_seq(ids=(1, 2, 3, 4), seq_id=2), k=4)
    assert set(p._premier_echec_amorcage) == {1, 2}


def test_model_mtp_prefill_releves_porte_seq_ids():
    # vérifié via EngineStats : le relevé lui-même exige un ForwardBatch réel (coûteux à
    # jouet) — ici on vérifie que le champ existe et son contrat par un relevé simulé.
    stats = runner_mod.EngineStats()
    stats.source_modele = lambda: types.SimpleNamespace(
        _mtp_prefill_releves=[{"evenement": "ecriture", "taille": 78, "seq_ids": [42]}])
    assert stats.to_dict()["mtp_prefill_releves"] == [{"evenement": "ecriture", "taille": 78, "seq_ids": [42]}]


def test_engine_stats_premiers_echecs_amorcage():
    spec = types.SimpleNamespace(_premier_echec_amorcage={7: {"cause": "prefill_court", "seq_id": 7}})
    stats = runner_mod.EngineStats()
    stats.source_speculateur = lambda: spec
    assert stats.to_dict()["spec_premiers_echecs_amorcage"] == {7: {"cause": "prefill_court", "seq_id": 7}}


def test_mtp_amorcage_echecs_bornes_a_4():
    h = torch.zeros(16, 8)
    p = _mtp(mtp_hidden=h, mtp_prefill=None)
    for _ in range(6):
        p.propose(_seq(ids=(1, 2, 3, 4, 5)), k=4)
    assert len(p._amorcage_echecs) == 4   # deque(maxlen=4) : casse si la borne disparaît


def test_model_mtp_prefill_releves_init_vide_et_borne():
    m = _jouet(0)
    assert list(m._mtp_prefill_releves) == []
    assert m._mtp_prefill_releves.maxlen == 4


def test_engine_stats_releves_amorcage_et_mtp_prefill():
    spec = types.SimpleNamespace(_amorcage_echecs=[{"cause": "prefill_absent"}])
    modele = types.SimpleNamespace(_mtp_prefill_releves=[{"evenement": "ecriture", "taille": 3}])
    stats = runner_mod.EngineStats()
    stats.source_speculateur = lambda: spec
    stats.source_modele = lambda: modele
    d = stats.to_dict()
    assert d["spec_amorcage_echecs"] == [{"cause": "prefill_absent"}]
    assert d["mtp_prefill_releves"] == [{"evenement": "ecriture", "taille": 3}]


# -- correctif C (01/10) : un pas qui émet un jeton nourrit _mtp_prefill ----
# chef, 2e tour : posé d'abord dans _plain_decode_sync, SANS EFFET (ce
# chemin n'est jamais emprunté avant le 1er essai spéculatif de nos vraies
# séquences — graphe, repli de la garde, chemin non isolé précisément).
# Déplacé au point commun à TOUT chemin qui émet un jeton : _consommer, juste
# après seq.output_ids.append. `model._mtp_hidden` posé directement (sans
# appeler `_garder_hidden`) simule un pas REJOUÉ SOUS GRAPHE : seul le
# `copy_` capturé écrit ce tampon, aucun code Python ne tourne — le jouet
# n'a donc pas besoin de distinguer eager/graphe, exactement le point voulu.

def _modele_mtp_factice(prefill, prefill_seq_id, hidden, hidden_n):
    return types.SimpleNamespace(mtp=object(), _mtp_prefill=prefill, _mtp_prefill_seq_id=prefill_seq_id,
                                 _mtp_hidden=hidden, _mtp_hidden_n=hidden_n, _mtp_prefill_releves=[])


def _engine_nourrir(modele):
    eng = types.SimpleNamespace(model=modele)
    eng._nourrir_mtp_prefill = types.MethodType(runner_mod.Engine._nourrir_mtp_prefill, eng)
    return eng


def test_nourrir_mtp_prefill_etend_meme_sequence_pas_graphe():
    h = torch.ones(1, 8) * 9.0   # posé directement : simule le tampon rempli par un rejeu de graphe
    m = _modele_mtp_factice(torch.zeros(78, 8), 15, h, 1)
    eng = _engine_nourrir(m)
    eng._nourrir_mtp_prefill(_seq(seq_id=15, output_ids=[1, 2]))   # 2e jeton ou plus : pas le tout premier
    assert m._mtp_prefill.shape[0] == 79
    assert torch.equal(m._mtp_prefill[-1], h[0])
    assert m._mtp_prefill_releves[-1] == {"evenement": "extension_pas_simple", "taille": 79, "seq_ids": [15]}


def test_nourrir_mtp_prefill_refuse_premier_jeton_de_la_sequence():
    """Le tout premier jeton (celui du préfill lui-même) ne doit PAS étendre : sa ligne est
    déjà la dernière de _mtp_prefill, l'ajouter une 2e fois décalerait tout d'un cran."""
    m = _modele_mtp_factice(torch.zeros(78, 8), 15, torch.ones(1, 8), 1)
    eng = _engine_nourrir(m)
    eng._nourrir_mtp_prefill(_seq(seq_id=15, output_ids=[1]))   # un seul jeton : le premier
    assert m._mtp_prefill.shape[0] == 78 and m._mtp_prefill_releves == []


def test_nourrir_mtp_prefill_refuse_sequence_differente():
    m = _modele_mtp_factice(torch.zeros(78, 8), 15, torch.ones(1, 8), 1)
    eng = _engine_nourrir(m)
    eng._nourrir_mtp_prefill(_seq(seq_id=16, output_ids=[1, 2]))   # autre séquence que le préfill courant
    assert m._mtp_prefill.shape[0] == 78 and m._mtp_prefill_releves == []


class _TeteFactice:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, *a, **k):
        return None


def test_amorcage_reussit_apres_extension_dun_pas_graphe():
    """Bout en bout (chef, après le décalage de 2 trouvé sur acvram-qwen3.8-27b-nvfp4) :
    préfill suivi d'un pas (hidden posé sans _garder_hidden, comme un rejeu de graphe),
    amorçage ensuite réussi — casse si _nourrir_mtp_prefill disparaît ou cesse d'étendre."""
    H = 8
    cache = types.SimpleNamespace(cfg=types.SimpleNamespace(num_blocks=8))
    modele = types.SimpleNamespace(mtp=_TeteFactice(cache), _mtp_prefill=torch.zeros(3, H),
                                   _mtp_prefill_seq_id=1, _mtp_hidden=torch.ones(1, H), _mtp_hidden_n=1,
                                   _mtp_prefill_releves=[], embed_tokens=torch.randn(10, H))
    p = MTPProposer(modele, max_model_len=4096)
    eng = _engine_nourrir(modele)
    seq = _seq(ids=(1, 2, 3, 4, 5), seq_id=1, output_ids=[1, 2])   # prompt 3 + 2 générés : len(ids)-1=4 > 3
    assert p._amorcer(seq, _DraftState()) is False
    eng._nourrir_mtp_prefill(seq)   # le pas qui vient de produire le 2e jeton nourrit le préfill
    assert modele._mtp_prefill.shape[0] == 4
    assert p._amorcer(seq, _DraftState()) is True
