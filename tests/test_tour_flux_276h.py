"""276 h : deux tours de vision en vol (deux flux annexes, une image chacune) — VerrouCapture lecteurs/rédacteur.
Cassants : (1) deux lecteurs coexistent, un rédacteur les exclut et est exclu par eux, priorité au rédacteur ;
(2) deux `encoder_images` de deux fils tournent EN MÊME TEMPS (lecteurs_max ≥ 2) et rendent les mêmes traits que la série ;
(3) sans CUDA, pas de réserve de flux (None) ; la variable de régime est déclarée."""
from __future__ import annotations

import threading
import time

import torch

from acvram.engine.graphs import VerrouCapture
from acvram.engine.vision import TourVision


def test_verrou_lecteurs_coexistent_redacteur_seul():
    v = VerrouCapture()
    dedans, fin = [], threading.Event()

    def lecteur(i):
        with v.lecteur():
            dedans.append(i)
            fin.wait(2)
    fils = [threading.Thread(target=lecteur, args=(i,)) for i in range(2)]
    for f in fils:
        f.start()
    for _ in range(50):
        if len(dedans) == 2:
            break
        time.sleep(0.01)
    assert len(dedans) == 2 and v.lecteurs_max == 2, "deux lecteurs devraient être dedans en même temps"
    # rédacteur : attend la fin des lecteurs ; un lecteur arrivant pendant l'attente attend aussi (priorité)
    pris, tard = threading.Event(), []

    def redacteur():
        with v:
            pris.set()
            time.sleep(0.2)

    def lecteur_tardif():
        with v.lecteur():
            tard.append(time.perf_counter())
    r = threading.Thread(target=redacteur); r.start(); time.sleep(0.05)
    assert not pris.is_set(), "le rédacteur est entré alors que deux lecteurs tenaient le verrou"
    t = threading.Thread(target=lecteur_tardif); t.start(); time.sleep(0.05)
    assert not tard, "un lecteur est entré alors qu'un rédacteur attendait (priorité au rédacteur)"
    fin.set()
    for f in fils:
        f.join(2)
    assert pris.wait(2), "le rédacteur n'est pas entré après la sortie des lecteurs"
    t0 = time.perf_counter(); r.join(2); t.join(2)
    assert tard and tard[0] >= t0 - 0.01, "le lecteur tardif devait passer après le rédacteur"
    assert not v.locked()


def _moteur(converted, appels, delai=0.0):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256, enable_cuda_graphs=False)
    h = loaded.spec.hidden_size

    class _Sortie:
        def __init__(self, t):
            self.pooler_output = t

    def calcul(pv):
        appels.append(float(pv)); time.sleep(delai)
        n = int(pv)
        return _Sortie(torch.arange(n * h, dtype=torch.float32).reshape(n, h) * float(pv))
    engine.vision = TourVision(calcul, torch.device("cpu"), nom="factice")
    engine.graphs = type("G", (), {})()          # à sec : graphes absents, le verrou seul compte ici
    engine.graphs.verrou_capture = VerrouCapture()
    return engine


def test_deux_encoder_images_en_vol_en_meme_temps_et_au_bit(converted):
    engine = _moteur(converted, [], delai=0.3)
    images = [[(2, 5, torch.tensor(3.0), "sha-a")], [(1, 3, torch.tensor(2.0), "sha-b")]]
    serie = [engine.encoder_images(im) for im in images]
    res = [None, None]

    def un(i):
        res[i] = engine.encoder_images(images[i])
    t0 = time.perf_counter()
    fils = [threading.Thread(target=un, args=(i,)) for i in range(2)]
    for f in fils:
        f.start()
    for f in fils:
        f.join(5)
    dt = time.perf_counter() - t0
    assert engine.graphs.verrou_capture.lecteurs_max >= 2, "les deux tours n'ont pas été en vol ensemble"
    assert dt < 0.55, f"deux tours de 0,3 s ont pris {dt:.2f} s : sérialisées"
    for (e1, n1), (e2, n2) in zip(serie, res):
        assert n1 is None and n2 is None
        assert len(e1) == len(e2) == 1 and torch.equal(e1[0][2], e2[0][2])
    assert engine._reserve_flux_tour() is None            # sans CUDA : flux courant


def test_variable_de_regime_declaree():
    from acvram import regime
    noms = {v.nom for v in regime.VARIABLES}
    assert {"TOUR_PREPARATION", "TOUR_FLUX"} <= noms
