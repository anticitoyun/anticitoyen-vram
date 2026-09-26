"""Pièce 146 (1) : une séquence tronquée par budget KV épuisé DOIT être livrée à l'appelant, sur chaque chemin de
décodage. `_finish_budget_epuise` rend une `GenerationOutput(finished=True, finish_reason="length")` ; le chemin
synchrone la rendait, mais le pipeline (pipeline.py, amorce et suite) et le spéculatif (runner.py) la jetaient : la
séquence quittait le lot sans fin signalée, et `collect` (server/app.py) attendait sans fin la requête du client. Le
moteur étant `idle` sans séquence, une sortie remise « au pas suivant » ne partirait jamais : elle part dans le pas même."""

import types

import pytest
import torch


@pytest.fixture
def moteur(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=2)
    eng = Engine(loaded, None, max_batch_size=2, max_model_len=128, enable_cuda_graphs=False)
    eng.add_request(list(range(5, 25)), SamplingParams(temperature=0.0, max_tokens=50), request_id="r146")
    eng.step()                                   # préfill + premier jeton
    assert [s.request_id for s in eng.running] == ["r146"]
    eng._grow = lambda seq, extra=0: False       # budget KV épuisé au pas suivant
    return eng


def _fins(sorties):
    return [o for o in sorties if o.request_id == "r146" and o.finished]


def _verifier(eng, sorties):
    fins = _fins(sorties)
    assert len(fins) == 1, f"séquence tronquée livrée {len(fins)} fois (attendu : 1)"
    assert fins[0].finish_reason == "length"
    assert not eng.running and eng.idle
    assert eng.stats.sequences_tronquees_budget == 1


def test_chemin_synchrone(moteur):
    _verifier(moteur, moteur.step())


def test_chemin_pipeline(moteur):
    moteur.pipeline_actif = True
    moteur.graphs = types.SimpleNamespace(enabled=True, max_ql=1)   # jamais touché : toute la file est épuisée
    moteur._pipeline_pendiente = None
    _verifier(moteur, moteur.step())


def test_chemin_speculatif(moteur):
    from acvram.engine.speculative import Proposal

    moteur.speculator = types.SimpleNamespace(propose=lambda seq, k: Proposal([7] * k), release=lambda seq: None)
    moteur.spec_k = 2
    moteur._garde_spec = types.SimpleNamespace(eligible=lambda b: True, enregistrer=lambda *a: None)
    _verifier(moteur, moteur.step())
