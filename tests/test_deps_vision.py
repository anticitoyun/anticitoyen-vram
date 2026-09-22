"""Dépendances de la tour de vision (trou P3 du 21/09 : le .venv reconstruit par
install.sh n'avait pas transformers → tout modèle multimodal, et P3, échouaient ;
un utilisateur du .deb aurait eu le même échec). transformers + pillow sont
INDISPENSABLES à `engine/vision.py` (AutoModelForImageTextToText) et à
`server/chat.py` (PIL). Ce test cassant verrouille la CHAÎNE de déclaration qui
garantit qu'`import transformers` réussit après install.sh : (1) pyproject déclare
l'extra `vision` avec transformers ET pillow ; (2) install.sh l'installe par défaut ;
(3) `acvram doctor` les vérifie. Test de source (pas de pip/réseau) : lancer
install.sh en vrai est réservé au trou de charge disque."""
from __future__ import annotations

import pathlib
import tomllib

RACINE = pathlib.Path(__file__).resolve().parent.parent


def test_pyproject_declare_extra_vision_avec_transformers_et_pillow():
    data = tomllib.loads((RACINE / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert "vision" in extras, "l'extra 'vision' manque de pyproject"
    noms = " ".join(extras["vision"]).lower()
    assert "transformers" in noms, extras["vision"]
    assert "pillow" in noms, extras["vision"]


def test_install_sh_installe_l_extra_vision():
    src = (RACINE / "install.sh").read_text()
    # l'unique `pip install -e` d'acvram doit inclure l'extra vision
    lignes = [l for l in src.splitlines() if "pip install" in l and "-e" in l and ".[" in l]
    assert lignes, "install.sh n'a pas de `pip install -e '.[...]'`"
    assert all("vision" in l for l in lignes), \
        f"install.sh installe acvram sans l'extra vision : {lignes}"


def test_le_deb_installe_transformers_et_pillow():
    src = (RACINE / "tools" / "construire-deb.sh").read_text()
    assert "transformers" in src and "pillow" in src, \
        "le .deb n'installe pas transformers/pillow (tour de vision)"


def test_doctor_verifie_transformers_et_pil():
    src = (RACINE / "acvram" / "cli.py").read_text()
    # cmd_doctor doit tester les deux modules de la tour de vision
    assert '"transformers"' in src and '"PIL"' in src, \
        "acvram doctor ne vérifie pas transformers/PIL"
    assert '"pillow"' in src, "acvram doctor doit nommer le paquet pip « pillow » pour PIL"
