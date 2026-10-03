"""La CI de l'arbre public tourne sous carte.sh, qui pose ACVRAM_ARBRE sur l'arbre principal : sans la retirer, la garde
de l'arbre (acvram/__init__.py) refuse l'import depuis la copie publique et la sortie 0.7.18 a été refusée (03/10)."""
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent


SCRIPT = ICI / "outils" / "publier-github.sh"


@pytest.mark.skipif(not SCRIPT.exists(), reason="outil privé, filtré de l'arbre public")
def test_la_ci_publique_retire_acvram_arbre():
    texte = SCRIPT.read_text()
    ligne = next(l for l in texte.splitlines() if "ACVRAM_TESTS_SOUS_CHARGE=1" in l)
    assert "env -u ACVRAM_ARBRE" in ligne
