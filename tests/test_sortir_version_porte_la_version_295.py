"""Pièce 295 (utilisateur, 28/09) : README.md était resté à 0.6.38 alors que la 0.7.10
était sortie — personne ne le tenait à jour au fil des versions. `outils/sortir-version.sh`
refuse désormais (code 74, distinct du 65 de pyproject.toml) si README.md § État ne porte
pas la version qu'on sort, ou n'a aucune ligne « Version X.Y.Z. » du tout. Dépôt jetable,
même convention que la 281/285b (`_depot_jetable` importé d'ici, pas dupliqué)."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_sortir_version_281 import _depot_jetable, _lancer, V, VNUM  # noqa: E402


def test_refuse_si_readme_porte_une_autre_version(tmp_path):
    d = _depot_jetable(tmp_path, readme_version="0.7.2")
    r = _lancer(d, f"v{V}")
    assert r.returncode == 74, r.stderr
    assert "README.md porte 0.7.2" in r.stderr, r.stderr


def test_refuse_si_readme_n_a_aucune_ligne_version(tmp_path):
    d = _depot_jetable(tmp_path, readme_version=None)
    r = _lancer(d, f"v{V}")
    assert r.returncode == 74, r.stderr
    assert "aucune ligne" in r.stderr, r.stderr


def test_accepte_si_readme_porte_la_bonne_version(tmp_path):
    d = _depot_jetable(tmp_path)  # readme_version par défaut = VNUM, comme pyproject.toml
    r = _lancer(d, f"v{V}", "--simule")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "== 1." in r.stdout


def test_readme_verifie_avant_les_notes_de_release(tmp_path):
    """Ordre des refus : README (nouveau) avant les notes de release (66/73) — pas de
    fausse impression que les notes manquent quand c'est le README qui cloche."""
    d = _depot_jetable(tmp_path, readme_version="0.0.0", avec_notes=False)
    r = _lancer(d, f"v{V}")
    assert r.returncode == 74, r.stderr
