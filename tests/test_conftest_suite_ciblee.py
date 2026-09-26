"""Second filet, indépendant du verrou-mesure (celui-ci ne protège que les
worktrees qui l'ont déjà fusionné, REGLES §1) : 18/09, load 90/32 cœurs,
trois suites COMPLETES de pairs en parallèle pendant une fenêtre P1. Une
suite pytest NON CIBLÉE (aucun chemin/nodeid, aucun -k, aucun -m) refuse de
démarrer si load1 > nproc/4, sauf ACVRAM_TESTS_SOUS_CHARGE=1."""
import os

import pytest

import conftest


class _FauxConfig:
    def __init__(self, file_or_dir=(), keyword="", markexpr=""):
        self._vals = {"file_or_dir": list(file_or_dir), "keyword": keyword, "markexpr": markexpr}

    def getoption(self, nom):
        return self._vals[nom]


def test_chemin_explicite_est_cible():
    assert conftest._suite_ciblee(_FauxConfig(file_or_dir=["tests/test_engine.py"]))


def test_keyword_est_cible():
    assert conftest._suite_ciblee(_FauxConfig(keyword="test_engine"))


def test_markexpr_est_cible():
    assert conftest._suite_ciblee(_FauxConfig(markexpr="not slow"))


def test_rien_de_tout_ca_n_est_pas_cible():
    assert not conftest._suite_ciblee(_FauxConfig())


def test_suite_non_ciblee_sous_charge_refuse(monkeypatch):
    """Le test qui casse : load 90 sur 32 cœurs (l'incident réel du 18/09),
    aucun chemin/nodeid/-k/-m -- doit refuser, pas tourner."""
    monkeypatch.setattr(os, "getloadavg", lambda: (90.0, 80.0, 70.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.delenv("ACVRAM_TESTS_SOUS_CHARGE", raising=False)
    monkeypatch.setenv("ACVRAM_TESTS_PENDANT_MESURE", "1")   # isole du garde verrou-mesure
    with pytest.raises(pytest.UsageError, match="suite complète sous charge"):
        conftest.pytest_configure(_FauxConfig())


def test_suite_ciblee_sous_la_meme_charge_ne_refuse_pas(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (90.0, 80.0, 70.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setenv("ACVRAM_TESTS_PENDANT_MESURE", "1")
    monkeypatch.setenv("ACVRAM_TESTS_SANS_VERROU", "1")
    conftest.pytest_configure(_FauxConfig(file_or_dir=["tests/test_engine.py"]))   # ne lève pas


def test_acvram_tests_sous_charge_force_le_passage(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (90.0, 80.0, 70.0))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setenv("ACVRAM_TESTS_SOUS_CHARGE", "1")
    monkeypatch.setenv("ACVRAM_TESTS_PENDANT_MESURE", "1")
    monkeypatch.setenv("ACVRAM_TESTS_SANS_VERROU", "1")
    conftest.pytest_configure(_FauxConfig())   # ne lève pas


def test_suite_non_ciblee_charge_basse_ne_refuse_pas(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (2.0, 1.8, 1.5))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.delenv("ACVRAM_TESTS_SOUS_CHARGE", raising=False)
    monkeypatch.setenv("ACVRAM_TESTS_PENDANT_MESURE", "1")
    monkeypatch.setenv("ACVRAM_TESTS_SANS_VERROU", "1")
    conftest.pytest_configure(_FauxConfig())   # ne lève pas
