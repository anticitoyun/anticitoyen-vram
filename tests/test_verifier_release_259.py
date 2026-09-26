"""Pièce 259 : `outils/verifier-release.sh` sur une release SIMULÉE (dossier de faux fichiers, jamais gh, jamais /tmp,
jamais l'installation Flatpak) — le script rend TENU quand tout est là et lisible, et FAUX (fichier MANQUE, PKGBUILD
non renseigné) quand il manque quelque chose. Casse si un nom attendu change dans release.yml sans que le script suive."""
import hashlib
import os
import pathlib
import subprocess
import tarfile
import zipfile

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "verifier-release.sh"
V = "0.7.0"


def _release_simulee(dossier: pathlib.Path, *, sans_rpm: bool = False, pkgbuild_skip: bool = False, sommes: bool = True,
                     ancien_bundle: bool = False) -> None:
    dossier.mkdir(parents=True)
    # .deb réel (dpkg-deb -b d'une arborescence minimale) : --info et --contents doivent le lire
    racine = dossier / "deb-src"; (racine / "DEBIAN").mkdir(parents=True); (racine / "usr" / "bin").mkdir(parents=True)
    (racine / "DEBIAN" / "control").write_text(
        f"Package: acvram\nVersion: {V}\nArchitecture: amd64\nMaintainer: acvram\nDescription: simulation 259\n", encoding="utf-8")
    (racine / "usr" / "bin" / "acvram").write_text("#!/bin/sh\necho simulation\n", encoding="utf-8")
    # 266 b : 300 entrées de plus — un .deb réel en a 274 ; avec `grep -q` sous pipefail le tube rendait faux (SIGPIPE)
    (racine / "usr" / "share" / "acvram").mkdir(parents=True)
    for i in range(300):
        (racine / "usr" / "share" / "acvram" / f"f{i:03d}").write_text("x", encoding="utf-8")
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
    # 266 i : le Flatpak est un .flatpakref (dépôt OSTree gh-pages + extra-data), plus un bundle
    (dossier / f"acvram-{V}.flatpakref").write_text("[Flatpak Ref]\nName=io.github.anticitoyen.acvram\nBranch=master\n"
                                                    "Url=https://anticitoyun.github.io/anticitoyen-vram/flatpak\n", encoding="utf-8")
    if ancien_bundle:
        (dossier / f"acvram-{V}.flatpak").write_bytes(b"bundle casse de la v0.7.0")
    with zipfile.ZipFile(dossier / f"translations-{V}.zip", "w") as z:
        z.writestr("fr.json", "{}"); z.writestr("en.json", "{}")
    if sommes:                                     # 259 b : SHA256SUMS comme le job `sommes` (sha256sum -- *)
        lignes = []
        for f in sorted(p for p in dossier.iterdir() if p.is_file()):
            lignes.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}")
        (dossier / "SHA256SUMS").write_text("\n".join(lignes) + "\n", encoding="utf-8")


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
                    "aur : sha256sums renseigné", "translations : 2 fichiers", "flatpak : acvram-0.7.0.flatpakref",
                    "sha256 : SHA256SUMS vérifié"):
        assert attendu in r.stdout, attendu
    assert "MANQUE" not in r.stdout and "FAUX " not in r.stdout and "SAUTÉ   sha256" not in r.stdout


@pytest.mark.skipif(subprocess.run(["which", "dpkg-deb"], capture_output=True).returncode != 0, reason="dpkg-deb requis")
def test_une_somme_fausse_est_vue(hors_tmp):
    """Témoin 259 b : un fichier modifié après la publication de SHA256SUMS rend FAUX."""
    _release_simulee(hors_tmp / "altere")
    (hors_tmp / "altere" / f"aur-{V}.tar.gz").write_bytes(b"archive remplacee")
    r = _lancer(hors_tmp / "altere")
    assert r.returncode == 1 and "FAUX    sha256 : SHA256SUMS ne correspond pas" in r.stdout, r.stdout


@pytest.mark.skipif(subprocess.run(["which", "dpkg-deb"], capture_output=True).returncode != 0, reason="dpkg-deb requis")
def test_sans_sommes_publiees_le_script_le_dit(hors_tmp):
    _release_simulee(hors_tmp / "sans-sommes", sommes=False)
    r = _lancer(hors_tmp / "sans-sommes")
    assert r.returncode == 0 and "SAUTÉ   sha256 : aucune somme publiée" in r.stdout


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


@pytest.mark.skipif(subprocess.run(["which", "dpkg-deb"], capture_output=True).returncode != 0, reason="dpkg-deb requis")
def test_l_ancien_bundle_encore_joint_est_faux(hors_tmp):
    """Témoin 266 i : le job doit retirer acvram-<V>.flatpak (bundle > 2 Gio, sans extra-data) ; s'il reste, FAUX."""
    _release_simulee(hors_tmp / "bundle", ancien_bundle=True)
    r = _lancer(hors_tmp / "bundle")
    assert r.returncode == 1 and "FAUX    flatpak : l'ancien bundle" in r.stdout, r.stdout


def test_259d_un_fichier_sur_disque_hors_release_n_est_pas_juge(hors_tmp):
    """259 d : le dossier de téléchargement est un cache. Un fichier d'un téléchargement précédent (l'ancien bundle) n'est
    pas « joint » : ce que la release joint, c'est liste-release.txt. Hors --simule, l'intrus va dans hors-release/."""
    texte = SCRIPT.read_text(encoding="utf-8")
    assert 'liste-release.txt' in texte and 'hors-release' in texte
    assert 'grep -qxF "acvram-${V}.flatpak" "$DOSSIER/liste-release.txt"' in texte, "le bundle se juge sur la liste"
    assert 'mv -f "$f" "$DOSSIER/hors-release/"' in texte and "rm " not in texte.split("hors-release/")[1][:200]
    # en --simule le dossier fait foi : le témoin 266 i (bundle présent → FAUX) reste rouge
    _release_simulee(hors_tmp / "b2", ancien_bundle=True)
    assert _lancer(hors_tmp / "b2").returncode == 1


def test_259d_l_app_deja_installee_est_retiree_avant_l_install():
    """259 d : --reinstall ne traverse pas les remotes (« already installed » depuis acvram-origin quand le .flatpakref
    propose acvram) — l'installation dédiée est vidée de l'app avant install --from."""
    texte = SCRIPT.read_text(encoding="utf-8")
    i = texte.index('flatpak uninstall --user --noninteractive -y "$APP"')
    assert i < texte.index('flatpak install --user --noninteractive -y --reinstall --from')
    assert 'flatpak info --user "$APP" >/dev/null 2>&1 &&' in texte[i - 80:i]
