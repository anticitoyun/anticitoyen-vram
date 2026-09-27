"""276 g : la tour de vision dans le fil de préparation (Engine.encoder_images), hors de `_admit`, sans lot.
Cassants : (1) traits identiques (au bit) au chemin `_admit`, et `_admit` n'appelle plus la tour pour une séquence qui
porte ses traits ; (2) la tour hors du pas tourne SOUS le verrou de capture des graphes ; (3) des traits sans rapport
avec les plages de la requête sont refusés nommément."""
from __future__ import annotations

import torch

from acvram.engine.vision import TourVision


class _Sortie:
    def __init__(self, traits):
        self.pooler_output = traits


def _moteur(converted, appels):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    h = loaded.spec.hidden_size

    def calcul(pv):
        appels.append(float(pv))
        n = int(pv)
        return _Sortie(torch.arange(n * h, dtype=torch.float32).reshape(n, h) * float(pv))

    engine.vision = TourVision(calcul, torch.device("cpu"), nom="factice")
    engine.spec.raw = {**(getattr(engine.spec, "raw", None) or {}), "architectures": ["Gemma4ForConditionalGeneration"]}
    return engine


def _prefills(engine):
    vus = []
    orig = engine._build_batch

    def espion(seqs, prefill, limite=None):
        b = orig(seqs, prefill, limite)
        vus.append(b)
        return b
    engine._build_batch = espion
    return vus


def _tourner(engine):
    for _ in range(50):
        engine.step()
        if not engine.running and not engine.waiting:
            return
    raise AssertionError("le moteur n'a pas fini en 50 pas")


def test_traits_de_preparation_au_bit_et_admit_sans_tour(converted):
    from acvram.engine.sampler import SamplingParams
    appels = []
    engine = _moteur(converted, appels)
    params = SamplingParams(temperature=0.0, max_tokens=2)
    images = [(2, 5, torch.tensor(3.0), "sha-a"), (6, 8, torch.tensor(2.0), "sha-b")]
    # témoin : tour dans `_admit`
    vus = _prefills(engine)
    engine.add_request(list(range(1, 10)), params, "temoin", images=images)
    _tourner(engine)
    assert appels == [3.0, 2.0], appels
    temoin = [b for b in vus if b.is_prefill][0].images[0]
    # 276 g : tour dans le fil de préparation, puis `add_request(traits=…)`
    appels.clear(); vus.clear()
    traits = engine.encoder_images(images, "prep")
    assert appels == [3.0, 2.0], "encoder_images appelle la tour une fois par image, dans l'ordre des plages"
    appels.clear()
    engine.add_request(list(range(1, 10)), params, "prep", images=images, traits=traits)
    _tourner(engine)
    assert appels == [], "la tour a retourné dans `_admit` alors que les traits étaient posés"
    pre = [b for b in vus if b.is_prefill][0].images[0]
    assert len(pre) == len(temoin) == 2
    for (d1, f1, e1), (d2, f2, e2) in zip(pre, temoin):
        assert (d1, f1) == (d2, f2) and e1.dtype == e2.dtype == torch.bfloat16 and torch.equal(e1, e2)


def test_encoder_images_sous_le_verrou_de_capture(converted):
    appels = []
    engine = _moteur(converted, appels)
    import threading
    import types
    # à sec (graphes désactivés) `engine.graphs` est None : le contrat se teste avec le même attribut que GraphRunner
    engine.graphs = types.SimpleNamespace(verrou_capture=threading.Lock())
    verrou = engine.graphs.verrou_capture
    tenu = []
    orig = engine.vision._calcul

    def calcul(pv):
        tenu.append(verrou.locked())
        return orig(pv)
    engine.vision._calcul = calcul
    engine.encoder_images([(2, 5, torch.tensor(3.0), "sha-a")])
    assert tenu == [True], tenu
    assert not verrou.locked(), "verrou de capture non rendu après la tour"


def test_traits_sans_rapport_refuses(converted):
    import pytest
    from acvram.engine.sampler import SamplingParams
    engine = _moteur(converted, [])
    params = SamplingParams(temperature=0.0, max_tokens=1)
    traits = engine.encoder_images([(2, 5, torch.tensor(3.0), "sha-a")])
    with pytest.raises(ValueError, match="276 g"):
        engine.add_request(list(range(1, 10)), params, "x", images=[(3, 6, torch.tensor(3.0), "sha-c")], traits=traits)
    assert engine.encoder_images([]) is None
