"""Pièce 223 : __version__ (acvram/__init__.py), pyproject.toml et le premier titre de section du
CHANGELOG doivent porter la même version — sinon un paquet, un affichage ou une note de release
peut annoncer une version que le code ne tient plus."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import acvram

RACINE = Path(__file__).resolve().parent.parent


def test_version_module_et_pyproject_identiques():
    pyproject = tomllib.loads((RACINE / "pyproject.toml").read_text())
    assert acvram.__version__ == pyproject["project"]["version"]


def test_premiere_section_du_changelog_porte_la_version_courante():
    changelog = (RACINE / "CHANGELOG.md").read_text()
    m = re.search(r"^## (\d+\.\d+\.\d+)", changelog, re.MULTILINE)
    assert m, "aucune section de version en tête du CHANGELOG"
    assert m.group(1) == acvram.__version__, (
        f"CHANGELOG en tête annonce {m.group(1)}, __version__ vaut {acvram.__version__}"
    )
