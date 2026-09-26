"""Pièce 273 (décision chef 26/09) : les modules REQUIS manquants sont un ECHEC (code 1) ; ceux d'un extra optionnel
de pyproject (vision : transformers, pillow) sont une ALERTE qui nomme l'extra à installer, et le doctor reste à 0.
verif-070 : le Flatpak v0.7.0 rendait « ECHEC transformers est absent » et code 1 pour un extra."""
import builtins

import pytest

from acvram import cli


@pytest.fixture
def sans(monkeypatch):
    """Rend l'import des modules nommés impossible, le reste inchangé."""
    def poser(*absents):
        vrai = builtins.__import__
        def factice(name, *a, **k):
            if name in absents:
                raise ImportError(name)
            return vrai(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", factice)
    return poser


def test_273_extra_vision_absent_est_une_alerte_et_le_doctor_reste_a_0(sans, capsys):
    sans("transformers", "PIL")
    assert cli._doctor_modules() is True
    sortie = capsys.readouterr().out
    assert "vision indisponible (transformers, pillow absent) : pip install 'acvram[vision]'" in sortie
    assert "ECHEC" not in sortie


def test_273_un_requis_absent_reste_un_echec(sans, capsys):
    sans("jinja2")
    assert cli._doctor_modules() is False
    assert "ECHEC jinja2 est absent (pip install jinja2)" in capsys.readouterr().out


def test_273_vision_presente_est_ok(capsys):
    pytest.importorskip("transformers"); pytest.importorskip("PIL")
    assert cli._doctor_modules() is True
    assert "ok    vision (transformers, PIL)" in capsys.readouterr().out


def test_273_les_extras_du_doctor_sont_ceux_de_pyproject():
    import pathlib, tomllib
    extras = tomllib.loads((pathlib.Path(cli.__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    for extra, mods in cli._EXTRAS.items():
        declares = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in extras[extra]}
        assert {p.lower() for _, p in mods} <= declares, (extra, declares)
