"""`ignore_eos`, même sémantique que vLLM/llama-server
(poste7-harnais-egal-ignore-eos-18-09) : sans lui, un harnais comparatif qui
force `--ignore-eos` côté llama.cpp mais n'a pas d'équivalent HTTP ici fait
tomber le lot acvram sous b à chaque séquence qui atteint l'EOS avant les
autres — la cellule b=12 du harnais égal reste indécidable.
"""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams


def _moteur(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    return Engine(loaded, None, max_batch_size=1, max_model_len=256,
                 enable_cuda_graphs=False), loaded


def test_par_defaut_tout_jeton_eos_arrete_au_premier(converted):
    engine, loaded = _moteur(converted)
    engine._eos = set(range(loaded.spec.vocab_size))   # tout jeton est EOS
    produced = [o for o in engine.generate([5, 42, 7],
                SamplingParams(temperature=0.0, max_tokens=6))]
    assert len(produced) == 1
    assert produced[-1].finish_reason == "stop"


def test_ignore_eos_continue_jusqu_a_max_tokens(converted):
    engine, loaded = _moteur(converted)
    engine._eos = set(range(loaded.spec.vocab_size))   # tout jeton est EOS
    produced = [o for o in engine.generate([5, 42, 7],
                SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True))]
    assert len(produced) == 6
    assert produced[-1].finish_reason == "length"
    assert engine.stats.sequences_ignore_eos == 1
    assert "ignore_eos=1" in engine.regime_ligne()


def test_ignore_eos_n_annule_pas_les_stop_token_ids_explicites(converted):
    engine, loaded = _moteur(converted)
    premier = [o for o in engine.generate([5, 42, 7],
               SamplingParams(temperature=0.0, max_tokens=1))][0]
    tok0 = premier.token_ids[0]
    produced = [o for o in engine.generate([5, 42, 7],
                SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True,
                               stop_token_ids=[tok0]))]
    assert produced[-1].finish_reason == "stop"
    assert len(produced) == 1
