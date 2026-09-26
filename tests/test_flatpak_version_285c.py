"""Pièce 285 c (chef, 26/09) : sur la v0.7.4, verifier-release.sh a écrit « OK flatpak : installé …
(0.7.2) » — il avait installé l'ancien commit parce que GitHub Pages n'avait pas encore servi le
nouveau dépôt OSTree, et l'a quand même jugé OK. Un `flatpak` simulé (faux binaire dans le PATH) qui
répond toujours 0.7.2 doit faire refuser (code distinct 66), jamais un OK trompeur, après une attente
bornée par `flatpak update` (10 tentatives, sommeil réduit à 0 sous test via ACVRAM_ATTENTE_FLATPAK_S)."""
import os
import pathlib
import shutil
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "verifier-release.sh"
V = "0.7.4"


@pytest.fixture
def hors_tmp():
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "acvram" / "tests-285c"
    d = base / f"p{os.getpid()}"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _faux_flatpak(dossier: pathlib.Path, version_installee: str) -> dict:
    """Un `flatpak` bidon : répond toujours `version_installee`, jamais un vrai dépôt OSTree/réseau."""
    (dossier / "flatpak").write_text(f'''#!/usr/bin/env bash
case "$1 $2" in
  "remote-add") exit 0 ;;
esac
if [ "$1" = "info" ]; then
  echo "Version: {version_installee}"
  exit 0
fi
if [ "$1" = "uninstall" ]; then exit 0; fi
if [ "$1" = "install" ]; then exit 0; fi
if [ "$1" = "update" ]; then exit 0; fi
if [ "$1" = "run" ]; then exit 0; fi
exit 0
''', encoding="utf-8")
    (dossier / "flatpak").chmod(0o755)
    return {"PATH": f"{dossier}:{os.environ['PATH']}"}


def _lancer(dossier: pathlib.Path, env: dict) -> subprocess.CompletedProcess:
    full_env = dict(os.environ, **env, ACVRAM_ATTENTE_FLATPAK_S="0")
    return subprocess.run(["bash", str(SCRIPT), f"v{V}", "--simule", str(dossier)],
                          capture_output=True, text=True, timeout=60, env=full_env)


def test_flatpak_jamais_a_jour_refuse_avec_code_distinct(tmp_path, hors_tmp):
    hors_tmp.mkdir(parents=True)
    (hors_tmp / f"acvram-{V}.flatpakref").write_text(
        "[Flatpak Ref]\nName=io.github.anticitoyen.acvram\nBranch=master\n"
        "Url=https://anticitoyun.github.io/anticitoyen-vram/flatpak\n", encoding="utf-8")
    faux_bin = tmp_path / "bin"; faux_bin.mkdir()
    env = _faux_flatpak(faux_bin, "0.7.2")
    r = _lancer(hors_tmp, env)
    assert r.returncode == 66, (r.returncode, r.stdout, r.stderr)
    assert "VERDICT: FAUX" in r.stdout
    assert "0.7.2" in r.stdout and V in r.stdout


def test_flatpak_a_jour_apres_attente_rend_tenu_sur_ce_controle(tmp_path, hors_tmp):
    """Té moin : si `flatpak info` finit par rendre la bonne version (Pages a rattrapé), pas de refus 66 —
    seul un `flatpak update` sans effet doit refuser, pas l'attente elle-même."""
    hors_tmp.mkdir(parents=True)
    (hors_tmp / f"acvram-{V}.flatpakref").write_text(
        "[Flatpak Ref]\nName=io.github.anticitoyen.acvram\nBranch=master\n"
        "Url=https://anticitoyun.github.io/anticitoyen-vram/flatpak\n", encoding="utf-8")
    faux_bin = tmp_path / "bin"; faux_bin.mkdir()
    # compteur sur disque : la 3e interrogation de `info` rend enfin la bonne version.
    compteur = hors_tmp / "compteur-info"
    (faux_bin / "flatpak").write_text(f'''#!/usr/bin/env bash
case "$1 $2" in
  "remote-add") exit 0 ;;
esac
if [ "$1" = "info" ]; then
  N=$(cat "{compteur}" 2>/dev/null || echo 0)
  N=$((N + 1))
  echo "$N" > "{compteur}"
  if [ "$N" -ge 3 ]; then echo "Version: {V}"; else echo "Version: 0.7.2"; fi
  exit 0
fi
if [ "$1" = "uninstall" ] || [ "$1" = "install" ] || [ "$1" = "update" ] || [ "$1" = "run" ]; then exit 0; fi
exit 0
''', encoding="utf-8")
    (faux_bin / "flatpak").chmod(0o755)
    env = {"PATH": f"{faux_bin}:{os.environ['PATH']}"}
    r = _lancer(hors_tmp, env)
    assert r.returncode != 66, (r.returncode, r.stdout, r.stderr)
    assert f"flatpak : installé" in r.stdout
