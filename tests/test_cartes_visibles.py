"""CUDA_VISIBLE_DEVICES doit restreindre ce que le planificateur voit.

nvidia-smi enumere le materiel entier quoi qu'il arrive. Sans filtrage, un plan
batit sur ses reponses nomme des cartes que le processus ne possede pas, et le
chargement tombe sur « invalid device ordinal » — mesure du 8/09/2026, trois
sessions se partageant la machine.
"""
from dataclasses import replace

import pytest

from acvram.hardware import detect as D


@pytest.fixture
def deux_cartes(monkeypatch):
    a = D.Gpu(index=0, name="A", uuid="GPU-aaa", total_mem=32 << 30)
    b = D.Gpu(index=1, name="B", uuid="GPU-bbb", total_mem=12 << 30)
    monkeypatch.setattr(D, "_probe_gpus_smi", lambda: ([a, b], "580"))
    monkeypatch.setattr(D, "_probe_p2p", lambda n: [])
    return a, b


@pytest.mark.parametrize("valeur,noms", [
    (None, ["A", "B"]),
    ("0,1", ["A", "B"]),
    ("1", ["B"]),
    ("0", ["A"]),
    ("", []),
    ("GPU-bbb", ["B"]),
    ("1,7", ["B"]),          # CUDA s'arrete au premier identifiant invalide
    ("7,1", []),
])
def test_filtrage(monkeypatch, deux_cartes, valeur, noms):
    if valeur is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", valeur)
    assert [g.name for g in D.detect_rig().gpus] == noms


def test_les_ordinaux_sont_renumerotes(monkeypatch, deux_cartes):
    """La carte physique 1, seule visible, doit s'appeler cuda:0.

    Garder son index physique donnerait un plan juste sur le papier et faux a
    l'execution : c'est precisement l'erreur que ce filtrage repare.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus = D.detect_rig().gpus
    assert [g.index for g in gpus] == [0]
    assert gpus[0].uuid == "GPU-bbb"      # bien la seconde carte physique
