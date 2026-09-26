"""Pièce 266 d : aucune dépendance Python du Flatpak n'arrive en sdist — numpy (meson), tokenizers et pydantic-core (cargo)
ne se construisent pas dans le bac à sable sans réseau (v0.7.0, run 36223036314 : « Cannot import 'mesonpy' »). Gardes :
(1) sources-pypi.py télécharge avec --only-binary=:all: et refuse tout fichier qui n'est pas une roue ; (2) ses modules sont
tous des sources `file` .whl avec sha256 ; (3) release.yml n'appelle plus flatpak-pip-generator et passe par sources-pypi.py ;
(4) un fichier python3-modules.json ou torch-cu130.json présent dans packaging/flathub ne contient aucune source non-roue."""
import hashlib
import json
import pathlib
import re
import sys

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
sys.path.insert(0, str(FLATHUB))
import importlib.util
_spec = importlib.util.spec_from_file_location("sources_pypi", FLATHUB / "sources-pypi.py")
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)


def _resoudre_factice(fichier, index, sha):
    return f"https://files.pythonhosted.org/packages/xx/yy/{fichier}"


def test_les_modules_sont_des_roues_avec_sha256(tmp_path):
    for nom in ("numpy-2.3.0-cp312-cp312-manylinux_2_28_x86_64.whl", "tokenizers-0.22.0-cp312-abi3-manylinux_2_17_x86_64.whl",
                "jinja2-3.1.6-py3-none-any.whl", "torch-2.14.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl",
                "nvidia_cublas-13.0.0.19-py3-none-manylinux_2_27_x86_64.whl", "nvidia_ml_py-13.590.44-py3-none-any.whl"):
        (tmp_path / nom).write_bytes(nom.encode())
    mods = SP.modules_depuis_roues(str(tmp_path), resoudre=_resoudre_factice)
    noms = [m["name"] for m in mods]
    # torch et nvidia-cublas : servis par torch-cu130.json ; nvidia-ml-py (NVML) n'y est pas et doit rester
    assert noms == ["python3-jinja2", "python3-numpy", "python3-nvidia-ml-py", "python3-tokenizers"], noms
    for m in mods:
        (src,) = m["sources"]
        assert src["type"] == "file" and src["url"].endswith(".whl") and re.fullmatch(r"[0-9a-f]{64}", src["sha256"])
        assert "--no-deps" in m["build-commands"][0] and "--no-index" in m["build-commands"][0]
    assert mods[1]["sources"][0]["sha256"] == hashlib.sha256(b"numpy-2.3.0-cp312-cp312-manylinux_2_28_x86_64.whl").hexdigest()


def test_un_sdist_est_refuse(tmp_path):
    """Témoin : ce que flatpak-pip-generator livrait pour numpy."""
    (tmp_path / "numpy-2.3.0.tar.gz").write_bytes(b"sdist")
    with pytest.raises(ValueError, match="sdist"):
        SP.modules_depuis_roues(str(tmp_path), resoudre=_resoudre_factice)


def test_le_telechargement_exige_des_roues(monkeypatch):
    vu = {}
    monkeypatch.setattr(SP.subprocess, "check_call", lambda cmd: vu.setdefault("cmd", cmd))
    SP.telecharger(["numpy", "uvicorn[standard]"], "/nulle-part", "3.14")
    cmd = vu["cmd"]
    assert "--only-binary=:all:" in cmd and "--python-version" in cmd and "--platform" in cmd and cmd[-2:] == ["numpy", "uvicorn[standard]"]
    assert "--no-binary" not in " ".join(cmd) and "--no-deps" not in cmd, "les dépendances transitives doivent être résolues, en roues"


def test_release_yml_ne_passe_plus_par_flatpak_pip_generator():
    t = (RACINE / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "flatpak-pip-generator" not in t.replace("# 266 d : plus de flatpak-pip-generator", "") and "pipgen.py" not in t
    assert "sources-pypi.py" in t and "sources-torch.sh" in t


@pytest.mark.parametrize("fichier", ["python3-modules.json", "torch-cu130.json"])
def test_aucune_source_non_roue_dans_un_json_present(fichier):
    p = FLATHUB / fichier
    if not p.exists():
        pytest.skip(f"{fichier} généré en CI, absent du dépôt")
    for m in json.loads(p.read_text(encoding="utf-8"))["modules"]:
        for s in m.get("sources", []):
            assert s["type"] == "file" and s["url"].endswith(".whl"), (m["name"], s)
