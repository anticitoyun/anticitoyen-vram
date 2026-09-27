"""dut 27/09 : à froid, l'autotune Triton fait grandir la limite de pile CUDA (12 256 o/fil mesurés) et le pilote garde la
mémoire locale correspondante (2,7 Gio) → chauffe à 1,6 Gio libres, service mort au premier démarrage. La limite est ramenée
à 1 024 avant la première capture et sous le seuil de la garde. À sec : faux pilote (ctypes)."""
import ctypes

import pytest
import torch

import acvram.engine.graphs as G
from acvram.engine.graphs import GraphRunner


class FauxPilote:
    def __init__(self, pile):
        self.pile, self.poses = pile, []

    def cuCtxGetLimit(self, ptr, quoi):
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value = self.pile; return 0

    def cuCtxSetLimit(self, quoi, valeur):
        self.poses.append(valeur.value); self.pile = valeur.value; return 0


@pytest.fixture
def pilote(monkeypatch):
    f = FauxPilote(12256)
    monkeypatch.setattr(ctypes, "CDLL", lambda nom: f)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.delenv("ACVRAM_PILE_RENDUE", raising=False); monkeypatch.delenv("ACVRAM_PILE_OCTETS", raising=False)
    return f


def test_pile_ramenee_au_defaut(pilote):
    assert G._rendre_pile() == (12256, 1024) and pilote.poses == [1024]
    assert G._rendre_pile() == (1024, 1024) and pilote.poses == [1024]      # déjà au défaut : rien reposé


def test_temoin_coupe(pilote, monkeypatch):
    monkeypatch.setenv("ACVRAM_PILE_RENDUE", "0")
    assert G._rendre_pile() is None and pilote.poses == []


def test_garde_rend_la_pile_sous_le_seuil(pilote, monkeypatch):
    """Rien dans le cache PyTorch (réservé = alloué), 920 Mio libres : avant, capture refusée ; après, la pile est rendue,
    nouvelle photo, capture admise."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    gr = GraphRunner.__new__(GraphRunner); gr.abandon_capture = None
    photos = iter([{"libre": 920 * 2 ** 20, "total": 32 * 2 ** 30, "reserve": 26 * 2 ** 30, "alloue": 26 * 2 ** 30},
                   {"libre": 3700 * 2 ** 20, "total": 32 * 2 ** 30, "reserve": 26 * 2 ** 30, "alloue": 26 * 2 ** 30}])
    gr._photo_memoire = lambda: next(photos)
    assert gr._garde_capture((12, 1, 8, 0)) is None and pilote.poses == [1024]
