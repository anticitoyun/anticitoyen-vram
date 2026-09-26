"""Pièce 273 : le Flatpak embarque l'extra vision — transformers et pillow dans la liste donnée à sources-pypi.py par
release.yml, et dans python3-modules.json s'il a été généré (44 roues cp314/abi3/py3 résolues à sec le 26/09)."""
import json
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parents[1]


def test_273_release_yml_demande_transformers_et_pillow():
    texte = (RACINE / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    m = re.search(r"sources-pypi\.py --python \"\$PYV\" \\\n\s*([^\n]*)\)", texte)
    assert m, "appel de sources-pypi.py introuvable"
    assert "transformers" in m.group(1).split() and "pillow" in m.group(1).split(), m.group(1)


def test_273_python3_modules_json_porte_vision_s_il_existe():
    f = RACINE / "packaging" / "flathub" / "python3-modules.json"
    if not f.exists():
        return
    noms = {m["name"] for m in json.loads(f.read_text(encoding="utf-8"))["modules"]}
    assert {"python3-transformers", "python3-pillow"} <= noms, noms
