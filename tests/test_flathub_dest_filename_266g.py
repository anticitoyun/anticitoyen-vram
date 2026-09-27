"""Pièce 266 g : chaque source `file` de roue porte `dest-filename` = le nom de roue DÉCODÉ. L'URL de torch cu130 encode le
« + » de la version locale (`torch-2.14.0%2Bcu130-…whl`) ; sans dest-filename, flatpak-builder garde ce nom et pip, qui lit
la version dans le nom de fichier, ne trouve « aucune distribution » (run 36227976586). Gardes : aucun « % » dans un
dest-filename, chaque nom est une roue valide (packaging.utils.parse_wheel_filename) ; témoin : le nom encodé est refusé."""
import importlib.util
import json
import pathlib

import pytest
from packaging.utils import parse_wheel_filename

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
_spec = importlib.util.spec_from_file_location("sources_pypi", FLATHUB / "sources-pypi.py")
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)
URL = "https://download-r2.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl"


def test_dest_filename_est_le_nom_de_roue_decode():
    s = SP.source_de_roue("torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl", URL, "0" * 64)
    assert s["dest-filename"] == "torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl" and "%" not in s["dest-filename"]
    nom, version, _, _ = parse_wheel_filename(s["dest-filename"])
    assert nom == "torch" and str(version) == "2.14.0+cu130"
    assert s["url"] == URL and s["type"] == "file"


def test_le_nom_encode_est_decode_et_le_nom_invalide_refuse():
    """Témoin : ce que flatpak-builder aurait gardé (dernier segment de l'URL) ; et un nom qui n'est pas une roue."""
    s = SP.source_de_roue("torch-2.14.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl", URL, "0" * 64)
    assert s["dest-filename"] == "torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl"
    with pytest.raises(ValueError):
        SP.source_de_roue("torch-2.14.0.tar.gz", URL, "0" * 64)
    with pytest.raises(ValueError):
        SP.source_de_roue("pas-une-roue.whl", URL, "0" * 64)


def test_les_modules_generes_portent_dest_filename(tmp_path):
    for nom in ("numpy-2.5.3-cp314-cp314-manylinux_2_28_x86_64.whl", "jinja2-3.1.6-py3-none-any.whl"):
        (tmp_path / nom).write_bytes(nom.encode())
    mods = SP.modules_depuis_roues(str(tmp_path), resoudre=lambda f, i, s: "https://files.pythonhosted.org/x/" + f.replace("+", "%2B"), version="3.14")
    for m in mods:
        (s,) = m["sources"]
        assert "%" not in s["dest-filename"] and parse_wheel_filename(s["dest-filename"])


def test_sources_torch_utilise_source_de_roue():
    t = (FLATHUB / "sources-torch.sh").read_text(encoding="utf-8")
    # 266 g : source `file` par source_de_roue ; 266 i : source `extra-data` par source_extra_data — les deux décodent le nom
    assert ("_m.source_de_roue(" in t or "_t.source_extra_data(" in t) and '"type": "file", "url": url' not in t and '"type": "extra-data", "url"' not in t


@pytest.mark.parametrize("fichier", ["python3-modules.json", "torch-cu130.json"])
def test_aucun_dest_filename_encode_dans_un_json_present(fichier):
    p = FLATHUB / fichier
    if not p.exists():
        pytest.skip(f"{fichier} généré en CI, absent du dépôt")
    for m in json.loads(p.read_text(encoding="utf-8"))["modules"]:
        for s in m.get("sources", []):
            assert "%" not in s.get("dest-filename", "") and parse_wheel_filename(s["dest-filename"])
