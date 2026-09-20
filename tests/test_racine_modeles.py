"""outils/racine_modeles.py : ACVRAM_MODELES → ~/.config/acvram/modeles (même lecture que cli.py) → littéral
(20/09 : GNOME n'hérite pas de environment.d, les chaînes doivent voir le parc sans variable)."""
from __future__ import annotations

import importlib.util
import os
import pathlib

RACINE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "racine_modeles.py"


def _charge(monkeypatch, env, xdg):
    for k in ("ACVRAM_MODELES", "ACVRAM_MODELS_DIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    spec = importlib.util.spec_from_file_location("racine_modeles_test", RACINE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_la_variable_gagne(tmp_path, monkeypatch):
    m = _charge(monkeypatch, {"ACVRAM_MODELES": str(tmp_path)}, tmp_path / "xdg")
    assert m.MODELES == str(tmp_path)


def test_le_fichier_de_configuration_est_lu_comme_cli(tmp_path, monkeypatch):
    parc = tmp_path / "parc"; parc.mkdir()
    conf = tmp_path / "xdg" / "acvram"; conf.mkdir(parents=True)
    (conf / "modeles").write_text(f"# commentaire\n\n{tmp_path / 'absent'}\n{parc}\n")   # une ligne absente est sautée
    m = _charge(monkeypatch, {}, tmp_path / "xdg")
    assert m.MODELES == str(parc)


def test_sans_rien_le_litteral(tmp_path, monkeypatch):
    m = _charge(monkeypatch, {}, tmp_path / "xdg-vide")
    assert m.MODELES == m._DEFAUT and m._DEFAUT.startswith("/mnt/")
