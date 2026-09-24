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


def test_le_moteur_ne_lance_jamais_ninja(monkeypatch, tmp_path):
    monkeypatch.setenv("ACVRAM_MARLIN_CACHE", str(tmp_path))
    monkeypatch.setattr(MP, "_EXT", None)
    import torch.utils.cpp_extension as ce
    monkeypatch.setattr(ce, "load", lambda *a, **k: pytest.fail("ninja lancé par le moteur (compiler=False)"))
    assert MP.charger(compiler=False) is None and MP.COMPILE_ICI      # .so absent : repli nommé, pas de compilation
    so = MP.chemin_so()
    so.parent.mkdir(parents=True, exist_ok=True)
    so.write_bytes(b"faux .so")
    vus = []
    monkeypatch.setattr(torch.ops, "load_library", lambda chemin: vus.append(str(chemin)))
    assert MP.charger(compiler=False) is not None
    assert vus == [str(so)] and not MP.COMPILE_ICI
    monkeypatch.setattr(MP, "_EXT", None)


def test_la_copie_des_sources_est_complete_et_marquee(tmp_path):
    src = MP._sources_en_cache(tmp_path)
    attendus = {str(q.relative_to(MP.ICI)) for q in MP._fichiers_sources()}
    presents = {str(q.relative_to(src)) for q in src.rglob("*") if q.is_file() and q.name != ".complet"}
    assert presents == attendus and (src / ".complet").read_text() == MP.empreinte_sources()
