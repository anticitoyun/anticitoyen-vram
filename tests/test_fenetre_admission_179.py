"""Pièce 179 b : la fenêtre d'admission (`EngineService._attendre_les_arrivees`) s'ouvre en RAFALE seulement. Une
requête seule n'attend pas : c'est la condition qui a fait passer la fenêtre au défaut (TTFT solo inchangé). Deux
requêtes en file attendent que la file cesse de grossir. Bras cassant (prise) : condition « ≥ 2 en file » retirée → le
premier test casse. Processeur seul, moteur factice."""
import time

from acvram.server import app as A


class _Moteur:
    def __init__(self, n):
        self.running, self.waiting, self.max_batch_size = [], list(range(n)), 8


def _service(n):
    s = A.EngineService.__new__(A.EngineService)
    s.engine = _Moteur(n)
    return s


def test_une_requete_seule_n_attend_pas(monkeypatch):
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    s = _service(1)
    t0 = time.perf_counter()
    s._attendre_les_arrivees()
    assert time.perf_counter() - t0 < 0.002


def test_une_rafale_attend_la_fenetre(monkeypatch):
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    s = _service(3)
    t0 = time.perf_counter()
    s._attendre_les_arrivees()
    dt = time.perf_counter() - t0
    assert 0.005 <= dt < 0.05, dt


def test_file_pleine_n_attend_pas(monkeypatch):
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    s = _service(8)
    t0 = time.perf_counter()
    s._attendre_les_arrivees()
    assert time.perf_counter() - t0 < 0.002


def test_defaut_cinq_ms():
    assert A._FENETRE_ADMISSION_S == 0.005 or __import__("os").environ.get("ACVRAM_ADMISSION_FENETRE_MS")
