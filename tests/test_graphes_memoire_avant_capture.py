"""Instrument de mémoire autour de la première capture (chantier
gemma-4-26B-A4B, `chantier-gemma-capture-godet1-20-09`) : avant la PREMIÈRE
capture d'un moteur, `GraphRunner` photographie la VRAM (libre / réservé /
alloué), l'écrit au journal en une ligne et l'expose sous
`regime()['graphes_memoire_avant_capture']` ; même photo après un échec.

À sec : pas de carte. On pose un `GraphRunner` nu (sans `__init__`) et on
remplace les trois lectures de `torch.cuda` par des valeurs connues — le test
vérifie que la photo est PRISE, une seule fois, avec ces valeurs, et que sans
CUDA elle vaut `None` plutôt qu'un chiffre inventé (REGLES § 4).
"""
import torch

from acvram.engine import graphs as G
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine

GIO = 2 ** 30


def _runner_nu(device="cuda:0"):
    g = G.GraphRunner.__new__(G.GraphRunner)
    g.device = torch.device(device) if device else None
    g.captures = 0
    g.memoire_avant_capture = None
    g.memoire_apres_echec = None
    return g


def _cuda_factice(monkeypatch, libre, reserve, alloue, total=32 * GIO):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: reserve)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: alloue)


def test_photo_avant_premiere_capture_une_fois(monkeypatch, capsys):
    g = _runner_nu()
    _cuda_factice(monkeypatch, libre=int(0.37 * GIO), reserve=27 * GIO, alloue=15 * GIO)
    g._avant_premiere_capture()
    photo = g.memoire_avant_capture
    assert photo == {"libre": int(0.37 * GIO), "total": 32 * GIO,
                     "reserve": 27 * GIO, "alloue": 15 * GIO}
    ligne = capsys.readouterr().out
    assert "[graphe] mémoire avant capture : libre 0.37 Gio, réservé 27.00, alloué 15.00" in ligne
    assert "cache non rendu 12.00" in ligne
    # seconde capture du même moteur : la photo ne bouge plus, rien n'est réimprimé
    _cuda_factice(monkeypatch, libre=9 * GIO, reserve=20 * GIO, alloue=15 * GIO)
    g.captures = 1
    g._avant_premiere_capture()
    assert g.memoire_avant_capture is photo
    assert capsys.readouterr().out == ""


def test_photo_apres_echec(monkeypatch, capsys):
    g = _runner_nu()
    _cuda_factice(monkeypatch, libre=100 * 2 ** 20, reserve=30 * GIO, alloue=16 * GIO)
    g._apres_echec_capture()
    assert g.memoire_apres_echec["reserve"] - g.memoire_apres_echec["alloue"] == 14 * GIO
    assert "[graphe] mémoire après l'échec : libre 0.10 Gio" in capsys.readouterr().out


def test_sans_cuda_la_photo_est_none_pas_un_chiffre(monkeypatch, capsys):
    """Témoin : à sec, aucune valeur n'est fabriquée et rien n'est imprimé."""
    g = _runner_nu()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    g._avant_premiere_capture()
    g._apres_echec_capture()
    assert g.memoire_avant_capture is None and g.memoire_apres_echec is None
    assert capsys.readouterr().out == ""
    assert _runner_nu(device=None)._photo_memoire() is None


def test_regime_porte_la_photo(converted):
    """`regime()` expose la photo du GraphRunner vivant ; None sans photo."""
    loaded = load_model(converted, dtype=torch.bfloat16, max_model_len=512,
                        device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=512,
                    enable_cuda_graphs=False)
    assert engine.regime()["graphes_memoire_avant_capture"] is None

    class _Faux:
        enabled = True
        raison = None
        memoire_avant_capture = {"libre": 1, "total": 4, "reserve": 3, "alloue": 2}
        memoire_apres_echec = None

    engine._graphes_demandes = True
    engine.graphs = _Faux()
    r = engine.regime()
    assert r["graphes_memoire_avant_capture"] == {"libre": 1, "total": 4, "reserve": 3, "alloue": 2}
    assert r["graphes_memoire_apres_echec"] is None
