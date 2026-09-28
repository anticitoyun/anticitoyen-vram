"""7x8b : Marlin PRÉCOMPILÉ (Flatpak : pas de nvcc) — chargé seulement s'il porte l'empreinte des sources installées.

Sans lui, le Flatpak servait tous les MoE nvfp4 sur la pile naturelle (repli nommé sur la ligne de régime seulement).
Tout ici tourne à sec : un .so factice porte ou non le marqueur ; `torch.ops.load_library` et `_compiler` sont relevés."""
import json
import pathlib

import pytest
import torch

from acvram.kernels import marlin_port as MP

RACINE = pathlib.Path(__file__).resolve().parent.parent


def _precompile(dossier: pathlib.Path, emp_json: str, emp_so: str, so_archs="sm_120f+sm_86+sm_89",
                **surcharge) -> pathlib.Path:
    from acvram.kernels import _abi_python
    cand = dossier / f"marlin-{MP.empreinte_sources()}"
    cand.mkdir(parents=True)
    (cand / "acvram_marlin.so").write_bytes(b"\x7fELF...." + MP.MARQUEUR + emp_so.encode() + b";archs="
                                            + so_archs.encode() + b"\0....")
    e = {"empreinte": emp_json, "archs": ["sm_120f", "sm_86", "sm_89"], "torch": torch.__version__,
         "cuda": str(torch.version.cuda), "python": _abi_python()}
    e.update(surcharge)
    (cand / "empreinte.json").write_text(json.dumps(e), encoding="utf-8")
    return cand / "acvram_marlin.so"


def _juger(dossier, caps=()):
    return MP.precompile_utilisable(dossier, MP.empreinte_sources(), caps, torch.__version__, torch.version.cuda)


def test_empreinte_juste_utilisable(tmp_path):
    emp = MP.empreinte_sources()
    so = _precompile(tmp_path, emp, emp)
    assert _juger(tmp_path, {(12, 0), (8, 6)}) == (str(so), "précompilé")


@pytest.mark.parametrize("cas", ["so_autre_empreinte", "json_autre_empreinte", "torch", "python", "arch",
                                 "so_autres_archs"])
def test_empreinte_fausse_refusee(tmp_path, cas):
    emp, autre = MP.empreinte_sources(), "000000000000"
    if cas == "so_autre_empreinte":
        _precompile(tmp_path, emp, autre)
    elif cas == "json_autre_empreinte":
        _precompile(tmp_path, autre, emp)
    elif cas == "torch":
        _precompile(tmp_path, emp, emp, torch="0.0.0")
    elif cas == "python":
        _precompile(tmp_path, emp, emp, python="cpython-399-x86_64-linux-gnu")
    elif cas == "arch":
        _precompile(tmp_path, emp, emp, so_archs="sm_86", archs=["sm_86"])
    else:                                    # 7x8c : la fiche promet sm_120f, le binaire n'a que sm_86
        _precompile(tmp_path, emp, emp, so_archs="sm_86")
    so, raison = _juger(tmp_path, {(12, 0)})
    assert so is None and raison != "précompilé", cas


@pytest.fixture
def charge(tmp_path, monkeypatch):
    """charger() à sec : dossier précompilé et cache Marlin temporaires ; load_library et _compiler relevés."""
    appels = {"load": [], "compiler": 0}
    monkeypatch.setenv("ACVRAM_KERNELS_PRECOMPILES", str(tmp_path / "pre"))
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(MP, "_EXT", None)
    monkeypatch.setattr(MP, "PRECOMPILE", None)
    monkeypatch.setattr(MP, "_SO_CHARGE", None)
    monkeypatch.setattr(torch.ops, "load_library", lambda p: appels["load"].append(str(p)))

    def compiler(verbose=False, caps=None):
        # un .so « compilé » dont le marqueur couvre exactement les cartes demandées
        appels["compiler"] += 1
        d = MP.dossier_cache(caps)
        d.mkdir(parents=True, exist_ok=True)
        (d / "acvram_marlin.so").write_bytes(MP.MARQUEUR + MP.empreinte_sources().encode() + b";archs="
                                             + MP._nom_archs(caps).encode() + b"\0")

    monkeypatch.setattr(MP, "_compiler", compiler)
    return tmp_path / "pre", appels


def test_charger_prend_le_precompile_d_empreinte_juste(charge):
    pre, appels = charge
    emp = MP.empreinte_sources()
    so = _precompile(pre, emp, emp)
    assert MP.charger() is not None
    assert appels == {"load": [str(so)], "compiler": 0}
    assert MP.PRECOMPILE == str(so)


def test_charger_refuse_l_empreinte_fausse_et_compile(charge):
    # Casse si charger() charge un précompilé sans vérifier le marqueur du .so
    pre, appels = charge
    _precompile(pre, MP.empreinte_sources(), "000000000000")
    with pytest.warns(UserWarning, match="Marlin précompilé refusé"):
        assert MP.charger() is not None
    assert appels["compiler"] == 1
    assert appels["load"] == [str(MP.dossier_cache() / "acvram_marlin.so")]
    assert MP.PRECOMPILE is None


def test_le_binaire_porte_le_marqueur_que_le_chargeur_cherche():
    # bindings.cpp écrit « acvram_marlin_empreinte=<MARLIN_PORT_EMPREINTE> », que _lancer_ninja pose en -D
    src = (RACINE / "acvram/kernels/marlin_port/bindings.cpp").read_text()
    assert (f'"{MP.MARQUEUR.decode()}" ACVRAM_MARLIN_STR(MARLIN_PORT_EMPREINTE) ";archs=" '
            'ACVRAM_MARLIN_STR(MARLIN_PORT_ARCHS)') in src
    py = (RACINE / "acvram/kernels/marlin_port/__init__.py").read_text()
    assert "-DMARLIN_PORT_EMPREINTE=" in py and "-DMARLIN_PORT_ARCHS=" in py


def test_la_ci_produit_et_verifie_marlin():
    assert "marlin_port.compiler_precompile" in (RACINE / "acvram/kernels/precompiles.py").read_text()
    for f in (".github/workflows/release.yml", "packaging/flathub/job-noyaux-precompiles.yml"):
        assert 'find build/noyaux -name acvram_marlin.so' in (RACINE / f).read_text(), f
    assert "Marlin charge (precompile " in (RACINE / "outils/verifier-release.sh").read_text()
    assert "Marlin charge (" in (RACINE / "acvram/cli.py").read_text()
