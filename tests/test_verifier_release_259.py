"""Pièce 259 : `outils/verifier-release.sh` sur une release SIMULÉE (dossier de faux fichiers, jamais gh, jamais /tmp,
jamais l'installation Flatpak) — le script rend TENU quand tout est là et lisible, et FAUX (fichier MANQUE, PKGBUILD
non renseigné) quand il manque quelque chose. Casse si un nom attendu change dans release.yml sans que le script suive."""
import os
import pathlib
import subprocess
import tarfile
import zipfile

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "verifier-release.sh"
V = "0.7.0"


def _release_simulee(dossier: pathlib.Path, *, sans_rpm: bool = False, pkgbuild_skip: bool = False) -> None:
    dossier.mkdir(parents=True)
    # .deb réel (dpkg-deb -b d'une arborescence minimale) : --info et --contents doivent le lire
    racine = dossier / "deb-src"; (racine / "DEBIAN").mkdir(parents=True); (racine / "usr" / "bin").mkdir(parents=True)
    (racine / "DEBIAN" / "control").write_text(
        f"Package: acvram\nVersion: {V}\nArchitecture: amd64\nMaintainer: acvram\nDescription: simulation 259\n", encoding="utf-8")
    (racine / "usr" / "bin" / "acvram").write_text("#!/bin/sh\necho simulation\n", encoding="utf-8")
    subprocess.run(["dpkg-deb", "-b", "--root-owner-group", str(racine), str(dossier / f"acvram_{V}_amd64.deb")],
                   check=True, capture_output=True)
    # AUR : acvram/PKGBUILD + acvram/.SRCINFO
    aur = dossier / "aur-src" / "acvram"; aur.mkdir(parents=True)
    somme = "SKIP" if pkgbuild_skip else "0" * 64
    (aur / "PKGBUILD").write_text(f"pkgname=acvram\npkgver={V}\npkgrel=1\nsha256sums=('{somme}')\n", encoding="utf-8")
    (aur / ".SRCINFO").write_text(f"pkgbase = acvram\n\tpkgver = {V}\n", encoding="utf-8")
    with tarfile.open(dossier / f"aur-{V}.tar.gz", "w:gz") as tz:
        tz.add(aur, arcname="acvram")
    if not sans_rpm:
        (dossier / f"acvram-{V}-1.fc42.noarch.rpm").write_bytes(b"rpm simule")
        (dossier / f"acvram-{V}-1.fc42.src.rpm").write_bytes(b"srpm simule")
    (dossier / f"acvram-{V}.flatpak").write_bytes(b"flatpak simule")
    with zipfile.ZipFile(dossier / f"translations-{V}.zip", "w") as z:
        z.writestr("fr.json", "{}"); z.writestr("en.json", "{}")


def _lancer(dossier: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), f"v{V}", "--simule", str(dossier), "--sans-flatpak"],
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def hors_tmp(tmp_path_factory):
    """Le script refuse /tmp : le dossier simulé vit sous le cache de l'utilisateur, nettoyé après."""
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "acvram" / "tests-259"
    d = base / f"p{os.getpid()}"
    yield d
    import shutil; shutil.rmtree(d, ignore_errors=True)


@pytest.mark.skipif(subprocess.run(["which", "dpkg-deb"], capture_output=True).returncode != 0, reason="dpkg-deb requis")
def test_release_complete_est_tenue(hors_tmp):
    _release_simulee(hors_tmp / "complete")
    r = _lancer(hors_tmp / "complete")
    assert r.returncode == 0 and "VERDICT: TENU" in r.stdout, r.stdout + r.stderr
    for attendu in ("deb : Package acvram", f"deb : Version {V}", "deb : usr/bin/acvram présent", f"aur : pkgver={V}",
                    "aur : sha256sums renseigné", "translations : 2 fichiers", "flatpak : acvram-0.7.0.flatpak"):
        assert attendu in r.stdout, attendu
    assert "MANQUE" not in r.stdout and "FAUX " not in r.stdout


@pytest.mark.skipif(subprocess.run(["which", "dpkg-deb"], capture_output=True).returncode != 0, reason="dpkg-deb requis")
def test_release_incomplete_est_fausse(hors_tmp):
    """Témoin : sans RPM et avec un PKGBUILD à sha256sums=('SKIP'), le script DOIT rendre FAUX."""
    _release_simulee(hors_tmp / "incomplete", sans_rpm=True, pkgbuild_skip=True)
    r = _lancer(hors_tmp / "incomplete")
    assert r.returncode == 1 and "VERDICT: FAUX" in r.stdout, r.stdout + r.stderr
    assert f"MANQUE  rpm : acvram-{V}-*.noarch.rpm" in r.stdout
    assert "FAUX    aur : sha256sums=('SKIP')" in r.stdout


def test_le_script_refuse_tmp():
    r = subprocess.run(["bash", str(SCRIPT), f"v{V}", "--simule", "/tmp/acvram-259"], capture_output=True, text=True)
    assert r.returncode == 64 and "/tmp" in r.stderr
