"""Le lien de soutien buymeacoffee.com/anticitoyen doit être présent PARTOUT où un
utilisateur ou un empaqueteur le cherche (décision Maîtresse 21/09) : au moins huit
fichiers nommés, dont FUNDING.yml, le README, pyproject, le CLI, le .deb et la GUI.
Cassant : si un de ces points de contact perd le lien, le compte tombe sous 8."""
from __future__ import annotations

import pathlib
import subprocess

RACINE = pathlib.Path(__file__).resolve().parent.parent
LIEN = "buymeacoffee.com/anticitoyen"

# points de contact indispensables (chemin relatif depuis la racine du dépôt)
ATTENDUS = [
    ".github/FUNDING.yml",
    "README.md",
    "pyproject.toml",
    "acvram/cli.py",
    "tools/construire-deb.sh",
    "parc/bin/claude-modeles",
    "parc/bin/kimi-modeles",
    "REPRISE.md",
    "CLAUDE.md",
]


def test_chaque_point_de_contact_porte_le_lien():
    manquants = [f for f in ATTENDUS if LIEN not in (RACINE / f).read_text()]
    assert not manquants, f"lien de soutien absent de : {manquants}"


def test_au_moins_huit_fichiers_suivis_portent_le_lien():
    r = subprocess.run(["git", "grep", "-l", LIEN], cwd=RACINE,
                       capture_output=True, text=True)
    fichiers = [l for l in r.stdout.splitlines() if l.strip()]
    assert len(fichiers) >= 8, f"seulement {len(fichiers)} fichiers suivis : {fichiers}"
