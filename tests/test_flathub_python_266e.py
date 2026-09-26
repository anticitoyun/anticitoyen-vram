"""Pièce 266 e : les roues du Flatpak, les chemins du manifeste et le .so précompilé sont ceux du Python du RUNTIME
(PYTHON_RUNTIME = 3.14 pour GNOME 51), pas du Python de la machine qui les prépare. La v0.7.0 prenait des roues cp312 :
le pip 3.14 du bac à sable ne trouvait pas httptools (run 36225015692). Gardes : (1) PYTHON_RUNTIME est une version X.Y ;
(2) les chemins python3.X du manifeste en sont ; (3) sources-pypi.py / sources-torch.sh la prennent par défaut et refusent
une roue d'une autre ABI (cp312, cp314t) ; (4) release.yml vérifie PYTHON_RUNTIME contre le SDK réel et la passe aux deux
scripts, et le job noyaux-precompiles compile sous ce Python ; (5) le chargeur refuse un .so précompilé d'une autre ABI."""
import importlib.util
import pathlib
import re

import pytest
import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
RELEASE = RACINE / ".github" / "workflows" / "release.yml"
_spec = importlib.util.spec_from_file_location("sources_pypi", FLATHUB / "sources-pypi.py")
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)


def test_python_runtime_est_une_version():
    v = (FLATHUB / "PYTHON_RUNTIME").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"3\.\d{1,2}", v) and SP.python_du_runtime() == v


def test_les_chemins_python_du_manifeste_sont_ceux_du_runtime():
    v = SP.python_du_runtime()
    t = (FLATHUB / "io.github.anticitoyen.acvram.yml").read_text(encoding="utf-8")
    versions = set(re.findall(r"/lib/python(3\.\d+)/", t))
    assert versions == {v}, f"manifeste : chemins python {versions}, runtime {v}"


@pytest.mark.parametrize("roue, version, attendu", [
    ("numpy-2.5.3-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "3.14", True),
    ("tokenizers-0.23.2-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "3.14", True),
    ("jinja2-3.1.6-py3-none-any.whl", "3.14", True),
    ("nvidia_cublas-13.0.0.19-py3-none-manylinux_2_27_x86_64.whl", "3.14", True),
    ("httptools-0.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.whl", "3.14", False),   # la roue de la v0.7.0
    ("torch-2.14.0+cu130-cp314-cp314t-manylinux_2_28_x86_64.whl", "3.14", False),                 # sans GIL : autre ABI
    ("torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl", "3.14", True),
    ("numpy-2.3.0.tar.gz", "3.14", False),
])
def test_roue_compatible(roue, version, attendu):
    assert SP.roue_compatible(roue, version) is attendu


def test_une_roue_d_une_autre_abi_est_refusee_par_le_generateur(tmp_path):
    (tmp_path / "httptools-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl").write_bytes(b"x")
    with pytest.raises(ValueError, match="incompatible"):
        SP.modules_depuis_roues(str(tmp_path), resoudre=lambda f, i, s: "https://x/" + f, version="3.14")


def test_le_telechargement_prend_la_version_du_runtime(monkeypatch):
    vu = {}
    monkeypatch.setattr(SP.subprocess, "check_call", lambda cmd: vu.setdefault("cmd", cmd))
    SP.telecharger(["numpy"], "/nulle-part", "3.14")
    cmd = vu["cmd"]
    assert cmd[cmd.index("--python-version") + 1] == "3.14" and "--implementation" in cmd


def test_sources_torch_prend_la_version_du_runtime():
    t = (FLATHUB / "sources-torch.sh").read_text(encoding="utf-8")
    assert 'PYV="${2:-$(cat "$(dirname "$0")/PYTHON_RUNTIME")}"' in t
    assert '"--python-version", pyv' in t and '--python-version "$PYV"' in t and "3.12" not in t.replace("cp<PYTHON_RUNTIME>", "")
    assert "roue_compatible" in t


def test_release_yml_verifie_le_sdk_et_passe_la_version():
    jobs = yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]
    run = "\n".join(s.get("run", "") for s in jobs["flatpak"]["steps"])
    assert "PYV=$(cat packaging/flathub/PYTHON_RUNTIME)" in run
    assert re.search(r'flatpak run --command=python3 org\.gnome\.Sdk//51 .*version_info', run), "le SDK réel doit être interrogé"
    assert '[ "$PYV" = "$SDK" ]' in run and 'sources-pypi.py --python "$PYV"' in run and 'sources-torch.sh 2.14.0 "$PYV"' in run
    noyaux = "\n".join(s.get("run", "") for s in jobs["noyaux-precompiles"]["steps"])
    assert "PYV=$(cat packaging/flathub/PYTHON_RUNTIME)" in noyaux and '"python$PYV" -m venv' in noyaux, \
        "le .so précompilé doit être compilé sous le Python du runtime (ABI CPython)"


def test_le_chargeur_refuse_un_so_d_une_autre_abi(tmp_path):
    """Témoin 266 e : un .so précompilé sous cp312 dans un Python 3.14 est refusé avec sa raison, pas importé."""
    import hashlib
    import json
    from acvram import kernels
    src = b"__global__ void k(){}"
    sha = hashlib.sha256(src).hexdigest()
    d = tmp_path / sha[:16]; d.mkdir()
    (d / "acvram_kernels.so").write_bytes(b"ELF" + (0x1234).to_bytes(8, "little"))
    base = {"src_sha": sha, "src_hash": "1234", "archs": ["sm_120f"], "torch": "2.14.0+cu130", "cuda": "13.0"}
    (d / "empreinte.json").write_text(json.dumps(dict(base, python="cpython-312-x86_64-linux-gnu")), encoding="utf-8")
    so, raison = kernels._precompile_utilisable(str(tmp_path), src, {(12, 0)}, "2.14.0+cu130", "13.0", python_abi="cpython-314-x86_64-linux-gnu")
    assert so is None and "ABI Python" in raison and "312" in raison
    (d / "empreinte.json").write_text(json.dumps(dict(base, python="cpython-314-x86_64-linux-gnu")), encoding="utf-8")
    so, raison = kernels._precompile_utilisable(str(tmp_path), src, {(12, 0)}, "2.14.0+cu130", "13.0", python_abi="cpython-314-x86_64-linux-gnu")
    assert so and raison == "précompilé"
    assert kernels._abi_python().startswith("cpython-")
