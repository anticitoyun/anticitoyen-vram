"""Pièce 266 i : le Flatpak se livre par DÉPÔT OSTree (gh-pages, flatpak/) + .flatpakref, torch et sa fermeture CUDA en
extra-data — un bundle unique ne porte pas les extra-data (« Extra data missing in detached metadata », prouvé) et pèserait
≈ 2 Gio, la limite d'un fichier de release. Gardes : sources extra-data complètes (filename décodé, url, sha256, size > 0,
only-arches) ; apply_extra dépaquette avec le python3 du runtime, sans réseau ni pip ; PYTHONPATH du manifeste vers /app/extra ;
release.yml : build-update-repo, gh-pages sous flatpak/ seulement, .flatpakref joint, ancien bundle retiré, GPG optionnelle avec
avertissement ; verifier-release.sh : .flatpakref, installation et doctor séparables."""
import importlib.util
import pathlib
import re

import pytest
import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
FLATHUB = RACINE / "packaging" / "flathub"
_spec = importlib.util.spec_from_file_location("sources_torch", FLATHUB / "sources_torch.py")
ST = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ST)
URL = "https://download-r2.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl"


def test_source_extra_data_complete_et_decodee():
    s = ST.source_extra_data("torch-2.14.0%2Bcu130-cp314-cp314-manylinux_2_28_x86_64.whl", URL, "0" * 64, 554622961)
    assert s == {"type": "extra-data", "filename": "torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl", "url": URL,
                 "sha256": "0" * 64, "size": 554622961, "only-arches": ["x86_64"]}
    with pytest.raises(ValueError):
        ST.source_extra_data("torch-2.14.0.tar.gz", URL, "0" * 64, 10)
    with pytest.raises(ValueError):
        ST.source_extra_data("torch-2.14.0+cu130-cp314-cp314-manylinux_2_28_x86_64.whl", URL, "0" * 64, 0)


def test_apply_extra_depaquette_sans_reseau_ni_pip():
    cmds = "\n".join(ST.APPLY_EXTRA)
    assert "python3 -m zipfile -e" in cmds and "site-packages" in cmds and "apply_extra.ok" in cmds
    assert "pip" not in cmds and "curl" not in cmds and "wget" not in cmds and cmds.startswith("set -e")


def test_sources_torch_sh_emet_des_extra_data_et_apply_extra():
    t = (FLATHUB / "sources-torch.sh").read_text(encoding="utf-8")
    assert "source_extra_data(" in t and '"type": "script", "dest-filename": "apply_extra"' in t and "_t.APPLY_EXTRA" in t
    assert "install -Dm755 apply_extra /app/bin/apply_extra" in t and "pip3 install" not in t


def test_le_manifeste_expose_les_extra_data_a_python():
    t = (FLATHUB / "io.github.anticitoyen.acvram.yml").read_text(encoding="utf-8")
    assert "--env=PYTHONPATH=/app/extra/site-packages" in t


def _flatpak_run() -> str:
    jobs = yaml.safe_load((RACINE / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))["jobs"]
    return "\n".join(s.get("run", "") for s in jobs["flatpak"]["steps"])


def test_release_yml_publie_un_depot_et_un_flatpakref_pas_un_bundle():
    run = _flatpak_run()
    assert "flatpak build-update-repo --generate-static-deltas" in run and "build-bundle" not in run
    assert re.search(r"cp -a repo pages/flatpak", run) and "rm -rf pages/flatpak" in run, "seul flatpak/ est touché sur gh-pages"
    assert "git -C pages add -A flatpak .nojekyll" in run, "aucun autre fichier de gh-pages n'entre dans le commit"
    assert 'gh release upload "$TAG" "acvram-$V.flatpakref"' in run and 'gh release delete-asset "$TAG" "acvram-$V.flatpak"' in run
    assert "[Flatpak Ref]" in run and "RuntimeRepo=https://flathub.org/repo/flathub.flatpakrepo" in run


def test_release_yml_gpg_optionnelle_avec_avertissement():
    run = _flatpak_run()
    assert 'if [ -n "${FLATPAK_GPG_KEY:-}" ]' in run and "--gpg-sign=" in run and "GPGKey=" in run
    assert "::warning::FLATPAK_GPG_KEY absent" in run and "--no-gpg-verify" in run
    jobs = yaml.safe_load((RACINE / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))["jobs"]
    assert any("FLATPAK_GPG_KEY" in str(s.get("env", {})) for s in jobs["flatpak"]["steps"])


def test_verifier_release_attend_le_flatpakref_et_separe_installation_et_doctor():
    t = (RACINE / "outils" / "verifier-release.sh").read_text(encoding="utf-8")
    assert 'un_seul "flatpak" "acvram-${V}.flatpakref"' in t and "--flatpak-installer" in t and "--flatpak-doctor" in t
    assert '--from "$FLAT"' in t and "--bundle" not in t and "l'ancien bundle" in t
