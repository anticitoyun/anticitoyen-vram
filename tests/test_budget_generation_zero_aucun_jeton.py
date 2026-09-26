"""Priorité 1 (chef, 24/09), suite : `token_budget()` (protocol.py) rend bien
0 pour `max_tokens=0`, mais `Engine._consommer` échantillonnait quand même un
jeton du prefill et l'ajoutait à `seq.output_ids` AVANT de constater
`1 >= 0` — `completion_tokens=1` malgré un budget nul (verdict-echo-propre.md).
Casse si le garde-fou `seq.params.max_tokens <= 0` est retiré de `_consommer`
(runner.py) : la séquence redevient à 1 jeton produit.
"""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams


def _moteur(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    return Engine(loaded, None, max_batch_size=1, max_model_len=256,
                 enable_cuda_graphs=False)


def test_max_tokens_zero_ne_produit_aucun_jeton(converted):
    engine = _moteur(converted)
    produced = [o for o in engine.generate([5, 42, 7],
                SamplingParams(temperature=0.0, max_tokens=0))]
    assert len(produced) == 1
    assert produced[-1].finish_reason == "length"
    assert produced[-1].completion_tokens == 0
    assert produced[-1].token_ids == []
