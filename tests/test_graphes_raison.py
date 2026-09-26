"""Un refus de graphes doit se nommer.

Le silence était le vrai défaut : sur un hybride, perdre les graphes coûte
11,9 % de débit, et rien ne le disait. Le banc mesurait alors un moteur
diminué en croyant mesurer le moteur.

Ces épreuves portent sur la CONSÉQUENCE — la raison est-elle posée, et
est-elle celle du bon cas — et non sur le texte du message.
"""
import os
import torch
import pytest

from acvram.engine.graphs import GraphRunner


class _FauxModele:
    """Assez de surface pour que _eligible aille jusqu'au test visé."""
    def __init__(self):
        self.layers = []
        self.caches = {}
        self.norm = type("N", (), {"weight": torch.zeros(1)})()
        self.lm_head = type("H", (), {"qweight": None})()

    def modules(self):
        return iter(())

    def named_modules(self):
        return iter(())


def test_raison_posee_quand_desactive_par_variable(monkeypatch):
    monkeypatch.setenv("ACVRAM_DISABLE_CUDA_GRAPHS", "1")
    monkeypatch.setenv("ACVRAM_GRAPHES_MUETS", "1")
    gr = GraphRunner(_FauxModele(), 2048)
    assert not gr.enabled
    assert "ACVRAM_DISABLE_CUDA_GRAPHS" in gr.raison


@pytest.mark.skipif(torch.cuda.is_available(), reason="épreuve du cas sans GPU")
def test_raison_posee_sans_gpu(monkeypatch):
    monkeypatch.delenv("ACVRAM_DISABLE_CUDA_GRAPHS", raising=False)
    monkeypatch.setenv("ACVRAM_GRAPHES_MUETS", "1")
    gr = GraphRunner(_FauxModele(), 2048)
    assert not gr.enabled and gr.raison


def test_construction_sans_cuda_ne_leve_pas_267(monkeypatch):
    """Pièce 267 (CI GitHub, roue torch CPU) : `self.evenement_jetons = torch.cuda.Event()`
    sans garde, à la construction, levait `RuntimeError: Tried to instantiate dummy base
    class Event` pour TOUT `GraphRunner` — même un moteur qui ne capturera jamais rien —
    dès que torch n'a pas d'extension CUDA (Event y est une classe factice). `is_available`
    simulé plutôt que dépendre de la machine qui joue le test (sur une carte réelle,
    `torch.cuda.Event()` réussit même simulé à False : le témoin utile est la CI CPU)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    gr = GraphRunner(_FauxModele(), 2048)
    assert gr.evenement_jetons is None


def test_la_raison_survit_a_l_initialisation(monkeypatch):
    """Épreuve du piège réel : `raison` posée après `_eligible` l'effaçait, et
    tous les cas nommés ressortaient en « raison non nommée »."""
    monkeypatch.setenv("ACVRAM_DISABLE_CUDA_GRAPHS", "1")
    monkeypatch.setenv("ACVRAM_GRAPHES_MUETS", "1")
    gr = GraphRunner(_FauxModele(), 2048)
    assert gr.raison != ""


def test_le_message_peut_etre_tu(monkeypatch, capsys):
    """Un test ou un banc qui lance mille chargements ne doit pas être noyé."""
    monkeypatch.setenv("ACVRAM_DISABLE_CUDA_GRAPHS", "1")
    monkeypatch.setenv("ACVRAM_GRAPHES_MUETS", "1")
    GraphRunner(_FauxModele(), 2048)
    assert "graphes CUDA désactivés" not in capsys.readouterr().out


def test_le_message_parle_par_defaut(monkeypatch, capsys):
    """Et l'inverse : sans la variable, il DOIT parler. Un détecteur qu'on n'a
    jamais vu parler n'est pas un détecteur."""
    monkeypatch.setenv("ACVRAM_DISABLE_CUDA_GRAPHS", "1")
    monkeypatch.delenv("ACVRAM_GRAPHES_MUETS", raising=False)
    GraphRunner(_FauxModele(), 2048)
    assert "graphes CUDA désactivés" in capsys.readouterr().out
