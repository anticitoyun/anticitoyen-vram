"""Pièce 161 (poste6, 24/09) : le cache du port Marlin est PAR EMPREINTE des sources, jamais le dossier partagé
d'avant (`~/.cache/acvram/marlin_port`), et le moteur (`charger(compiler=False)`) ne lance jamais ninja — il charge
le .so de son empreinte ou rend None (repli nommé). Tests à sec, sans nvcc ni carte."""
import pathlib

import pytest
import torch

from acvram.kernels import marlin_port as MP

PARTAGE_D_AVANT = pathlib.Path.home() / ".cache" / "acvram" / "marlin_port"


def test_le_cache_porte_l_empreinte_des_sources(monkeypatch):
    monkeypatch.delenv("ACVRAM_MARLIN_CACHE", raising=False)
    e = MP.empreinte_sources()
    d = MP.dossier_cache()
    assert len(e) == 12 and int(e, 16) >= 0
    assert d.name == f"marlin_port-{e}"
    assert d != PARTAGE_D_AVANT, "le .so est redevenu partagé entre les worktrees"


def test_une_autre_source_un_autre_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("ACVRAM_MARLIN_CACHE", raising=False)
    avant = MP.dossier_cache()
    f = tmp_path / "x.cu"
    f.write_bytes(b"// autre version\n")
    monkeypatch.setattr(MP, "_EMPREINTE", None)
    monkeypatch.setattr(MP, "_fichiers_sources", lambda: [f])
    monkeypatch.setattr(MP, "ICI", tmp_path)
    apres = MP.dossier_cache()
    assert apres != avant and apres.name.startswith("marlin_port-")


def test_la_racine_est_une_option_pas_le_dossier(monkeypatch, tmp_path):
    """ACVRAM_MARLIN_CACHE (observation, tests) est une RACINE : l'empreinte reste dans le nom."""
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    assert MP.dossier_cache() == tmp_path / f"marlin_port-{MP.empreinte_sources()}"


def test_empreinte_absente_compile_une_fois_puis_charge(monkeypatch, tmp_path):
    """Réserve de chef (Marlin = défaut) : cache vide → compilation UNE fois (sous flock) puis load_library ;
    empreinte présente → aucun ninja."""
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    monkeypatch.setattr(MP, "_EXT", None)
    monkeypatch.setattr(MP, "COMPILE_ICI", False)
    so = MP.chemin_so()
    compilations, charges = [], []

    def faux_ninja(cache, verbose, load):
        compilations.append(str(cache))
        so.write_bytes(b"faux .so")
    monkeypatch.setattr(MP, "_lancer_ninja", faux_ninja)
    monkeypatch.setattr(torch.ops, "load_library", lambda chemin: charges.append(str(chemin)))
    assert MP.charger(compiler=False) is not None
    assert compilations == [str(so.parent)] and charges == [str(so)] and MP.COMPILE_ICI
    assert (so.parent / ".verrou-compilation").exists()
    # empreinte présente : plus jamais ninja, ni ici ni sous compiler=True
    monkeypatch.setattr(MP, "_EXT", None)
    monkeypatch.setattr(MP, "_lancer_ninja", lambda *a: pytest.fail("ninja relancé alors que le .so de l'empreinte existe"))
    assert MP.charger(compiler=True) is not None and charges == [str(so)] * 2
    monkeypatch.setattr(MP, "_EXT", None)


def test_compilation_echouee_repli_nomme(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    monkeypatch.setattr(MP, "_EXT", None)
    monkeypatch.setattr(MP, "ECHEC_COMPILATION", None)

    def nvcc_absent(cache, verbose, load):
        raise RuntimeError("nvcc introuvable")
    monkeypatch.setattr(MP, "_lancer_ninja", nvcc_absent)
    monkeypatch.setattr(torch.ops, "load_library", lambda chemin: pytest.fail("chargement d'un .so absent"))
    assert MP.charger(compiler=False) is None
    assert MP.ECHEC_COMPILATION == "RuntimeError: nvcc introuvable"


def test_le_verrou_de_compilation_saute_si_un_autre_a_fini(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    monkeypatch.setattr(MP, "COMPILE_ICI", False)
    so = MP.chemin_so()
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(b"compile par un autre processus")
    monkeypatch.setattr(MP, "_lancer_ninja", lambda *a: pytest.fail("recompilation d'un .so présent"))
    MP._compiler()
    assert not MP.COMPILE_ICI


def test_la_copie_des_sources_est_complete_et_marquee(tmp_path):
    src = MP._sources_en_cache(tmp_path)
    attendus = {str(q.relative_to(MP.ICI)) for q in MP._fichiers_sources()}
    presents = {str(q.relative_to(src)) for q in src.rglob("*") if q.is_file() and q.name != ".complet"}
    assert presents == attendus and (src / ".complet").read_text() == MP.empreinte_sources()
