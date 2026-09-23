"""Garde symétrique de celle de `pytest_configure` (18/09, load 70,8,
REGLES §2, sage-charge-cpu-verrou-mesure-18-09) : trois `pytest` de pairs
ont tourné pendant une fenêtre HTTP mesurée sans qu'aucun ne le voie, parce
que `_gpu_demande` ne regarde rien sans `CUDA_VISIBLE_DEVICES` (le défaut
de session, vide). Cette garde-ci lit `$VERROU.qui` sans condition."""
import os
import time

import pytest

import conftest


def _ecrire_verrou(tmp_path, type_="mesure", pid=None, nom="campagne-jumelle"):
    (tmp_path / "acvram-carte-0.lock").touch()   # glob.glob le cherche, seul .qui est lu
    (tmp_path / "acvram-carte-0.lock.qui").write_text(
        f"{pid or os.getpid()} {int(time.time())} {nom} {type_}\n")
    return str(tmp_path / "acvram-carte-0.lock")


def test_verrou_mesure_vivant_est_detecte(tmp_path, monkeypatch):
    _ecrire_verrou(tmp_path)
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    tenue = conftest._verrou_tenu_en_mesure()
    assert tenue is not None
    verrou, pid, nom = tenue
    assert pid == os.getpid()
    assert nom == "campagne-jumelle"


def test_pytest_configure_refuse_sous_faux_verrou_mesure(tmp_path, monkeypatch):
    """Le test qui casse : un faux verrou en TYPE=mesure, puis
    `pytest_configure` — doit refuser, pas tourner."""
    _ecrire_verrou(tmp_path)
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    monkeypatch.delenv("ACVRAM_TESTS_PENDANT_MESURE", raising=False)
    monkeypatch.setenv("ACVRAM_TESTS_SOUS_CHARGE", "1")   # ne pas dépendre de la charge réelle ici
    with pytest.raises(pytest.UsageError, match="TYPE=mesure"):
        conftest.pytest_configure(None)


def test_acvram_tests_pendant_mesure_force_le_passage(tmp_path, monkeypatch):
    _ecrire_verrou(tmp_path)
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    monkeypatch.setenv("ACVRAM_TESTS_PENDANT_MESURE", "1")
    monkeypatch.setenv("ACVRAM_TESTS_SANS_VERROU", "1")   # ne pas prendre le vrai flock ensuite
    monkeypatch.setenv("ACVRAM_TESTS_SOUS_CHARGE", "1")   # ne pas dépendre de la charge réelle ici
    conftest.pytest_configure(None)   # ne lève pas


def test_type_etat_n_invalide_pas(tmp_path, monkeypatch):
    _ecrire_verrou(tmp_path, type_="etat")
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    assert conftest._verrou_tenu_en_mesure() is None


def test_detenteur_disparu_est_ignore(tmp_path, monkeypatch):
    _ecrire_verrou(tmp_path, pid=2 ** 30 - 1)   # PID quasi sûrement mort
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    assert conftest._verrou_tenu_en_mesure() is None
