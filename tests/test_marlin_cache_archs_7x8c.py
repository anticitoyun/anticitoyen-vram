"""7x8c : le cache JIT de marlin_port est par empreinte des sources ET par architectures ; le .so déclare ses archs
dans son marqueur (bindings.cpp) et n'est chargé que s'il couvre les cartes. Le poste a deux cartes (5090 sm_120,
3080 Ti sm_86) : « les cibles sont constantes sur un poste » (161) était faux. À sec : .so factices."""
import pytest
import torch

from acvram.kernels import marlin_port as MP

_COMPILER = MP._compiler


def _so(caps_nom: str, archs: str):
    """Un .so factice dans le dossier de cache « caps_nom », dont le marqueur déclare « archs »."""
    d = MP._racine_cache() / f"marlin_port-{MP.empreinte_sources()}-{caps_nom}"
    d.mkdir(parents=True, exist_ok=True)
    so = d / "acvram_marlin.so"
    so.write_bytes(b"\x7fELF" + MP.MARQUEUR + MP.empreinte_sources().encode() + b";archs=" + archs.encode() + b"\0..")
    return so


@pytest.fixture
def poste(tmp_path, monkeypatch):
    appels = {"load": [], "compiler": []}
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    monkeypatch.setenv("ACVRAM_KERNELS_PRECOMPILES", str(tmp_path / "aucun"))
    monkeypatch.delenv("ACVRAM_ARCHS", raising=False)
    for nom in ("_EXT", "PRECOMPILE", "_SO_CHARGE"):
        monkeypatch.setattr(MP, nom, None)
    monkeypatch.setattr(torch.ops, "load_library", lambda p: appels["load"].append(str(p)))

    def compiler(verbose=False, caps=None):
        appels["compiler"].append(MP._nom_archs(caps))
        _so(MP._nom_archs(caps), "+".join(f"sm_{a}{b}{'f' if a >= 10 else ''}" for a, b in sorted(caps)))
    monkeypatch.setattr(MP, "_compiler", compiler)
    return monkeypatch, appels


def test_cache_sm86_jamais_charge_pour_sm120(poste):
    # Casse si la clé ou le chargement ignore les architectures (clé d'avant : empreinte seule)
    mp, appels = poste
    mp.setattr(MP, "_caps_requises", lambda: {(12, 0)})
    faux = _so("sm_86", "sm_86")
    assert MP.charger() is not None
    assert str(faux) not in appels["load"]
    assert appels["compiler"] == ["sm_120"] and appels["load"] == [str(MP.dossier_cache({(12, 0)}) / "acvram_marlin.so")]


def test_nom_de_dossier_menteur_refuse(poste):
    # Le dossier dit sm_120, le binaire déclare sm_86 : le binaire fait foi
    mp, appels = poste
    mp.setattr(MP, "_caps_requises", lambda: {(12, 0)})
    menteur = _so("sm_120", "sm_86")
    menteur.parent.rename(menteur.parent.with_name(menteur.parent.name + "-x"))   # hors du dossier exact
    _so("sm_120", "sm_86")
    assert MP.charger() is not None
    assert appels["compiler"] == ["sm_120"]


def test_une_carte_reutilise_le_so_deux_cartes_sans_compiler(poste):
    # Compilé à sec pour le poste (sm_86 + sm_120f) : une prise qui ne voit que la 5090 le prend, sans nvcc sous verrou
    mp, appels = poste
    deux = _so("sm_86+sm_120", "sm_120f+sm_86")
    mp.setattr(MP, "_caps_requises", lambda: {(12, 0)})
    assert MP.charger() is not None
    assert appels == {"load": [str(deux)], "compiler": []}


def test_deux_cartes_ne_prennent_pas_un_so_une_carte(poste):
    mp, appels = poste
    _so("sm_120", "sm_120f")
    mp.setattr(MP, "_caps_requises", lambda: {(8, 6), (12, 0)})
    assert MP.charger() is not None
    assert appels["compiler"] == ["sm_86+sm_120"]


def test_so_d_avant_7x8c_sans_archs_recompile(poste):
    # Un .so sans archs dans son marqueur (binaire d'avant) n'est ni chargé ni pris pour « déjà compilé »
    mp, appels = poste
    import acvram.kernels as K
    mp.setattr(MP, "_compiler", _COMPILER)
    mp.setattr(K, "_arch_flags", lambda archs_forcees=None, **k: [f"-gencode=arch=compute_{a}{b},code=sm_{a}{b}"
                                                                   for a, b in archs_forcees])
    d = MP.dossier_cache({(12, 0)})
    d.mkdir(parents=True)
    (d / "acvram_marlin.so").write_bytes(MP.MARQUEUR + MP.empreinte_sources().encode() + b"\0")
    assert MP.archs_du_so(d / "acvram_marlin.so", MP.empreinte_sources()) is None
    ninja = []

    def faux_ninja(cache, verbose, load, arch_flags):
        ninja.append(arch_flags)
        (cache / "acvram_marlin.so").write_bytes(MP.MARQUEUR + MP.empreinte_sources().encode() + b";archs=sm_120\0")
    mp.setattr(MP, "_lancer_ninja", faux_ninja)
    mp.setattr(MP, "_caps_requises", lambda: {(12, 0)})
    assert MP.charger() is not None
    assert ninja == [["-gencode=arch=compute_120,code=sm_120"]] and appels["load"] == [str(d / "acvram_marlin.so")]


def test_la_cle_porte_les_archs(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    assert MP.dossier_cache({(12, 0), (8, 6)}).name == f"marlin_port-{MP.empreinte_sources()}-sm_86+sm_120"
    assert MP.dossier_cache({(12, 0)}) != MP.dossier_cache({(8, 6)})
