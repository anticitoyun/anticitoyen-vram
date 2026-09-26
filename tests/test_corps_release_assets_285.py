"""Pièce 285 (chef, 26/09) : les notes de la release v0.7.3 citaient
`.../v0.7.3/acvram-0.7.3.flatpakref`, un fichier qui n'a jamais existé (le job flatpak avait
échoué, 266 m) — rien ne l'a signalé, `outils/verifier-release.sh` restait TENU. Choix :
ajouté à `verifier-release.sh` (pas à l'attente de `sortir-version.sh`) — `sortir-version.sh`
délègue déjà toute vérification post-publication à ce script (étape 7, pièce 281) ; dupliquer
la logique aurait donné deux sources de vérité sur ce qu'une release doit contenir. Un asset
cité dans le corps mais absent des assets joints refuse avec un code DISTINCT (65), jamais
confondu avec un MANQUE générique (1). Sans réseau : `--simule`, aucun `gh` appelé."""
import os
import pathlib
import subprocess
import sys

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "verifier-release.sh"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_verifier_release_259 import _release_simulee, V as V_259  # noqa: E402


@pytest.fixture
def hors_tmp():
    """Le script refuse /tmp (259) : dossier simulé hors /tmp, nettoyé après."""
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "acvram" / "tests-285"
    d = base / f"p{os.getpid()}"
    yield d
    import shutil; shutil.rmtree(d, ignore_errors=True)


def _lancer(dossier: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), f"v{V_259}", "--simule", str(dossier), *args],
                          capture_output=True, text=True, timeout=120)


def test_cas_0_7_3_asset_cite_absent_rend_faux_code_distinct(hors_tmp):
    """Reproduit exactement l'incident : le corps cite le .flatpakref, il n'est joint nulle
    part (job flatpak échoué) — aucun autre fichier n'a besoin d'exister, le refus est net
    et immédiat, avant même les autres contrôles."""
    d = hors_tmp / "v0.7.3"
    d.mkdir(parents=True)
    (d / "corps-release.txt").write_text(
        "Installer : `flatpak install --user "
        f"https://github.com/anticitoyun/anticitoyen-vram/releases/download/v{V_259}/"
        f"acvram-{V_259}.flatpakref`\n", encoding="utf-8")
    # aucun acvram-<V>.flatpakref sur le disque : c'est exactement l'incident du 26/09.
    r = _lancer(d, "--sans-flatpak")
    assert r.returncode == 65, (r.returncode, r.stdout, r.stderr)
    assert "VERDICT: FAUX" in r.stdout
    assert f"acvram-{V_259}.flatpakref" in r.stdout
    assert "corps de release" in r.stdout


def test_cas_complet_tous_les_assets_cites_sont_presents_rend_tenu(hors_tmp):
    """La release complète (fixture 259) + un corps qui cite exactement ce qui est joint :
    aucun refus, le script continue jusqu'à TENU."""
    d = hors_tmp / "complete"
    _release_simulee(d)
    cites = "\n".join(
        f"- https://github.com/anticitoyun/anticitoyen-vram/releases/download/v{V_259}/{p.name}"
        for p in sorted(d.iterdir()) if p.is_file())
    (d / "corps-release.txt").write_text(f"Assets :\n{cites}\n", encoding="utf-8")
    r = _lancer(d, "--sans-flatpak")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "VERDICT: TENU" in r.stdout
    assert "corps de release : tous les assets cités sont joints" in r.stdout


def test_sans_corps_release_le_controle_est_saute_pas_fatal(hors_tmp):
    """Une release simulée sans corps-release.txt (cas des autres tests 259, écrits avant
    cette pièce) ne doit pas se mettre à échouer : le contrôle se saute proprement."""
    d = hors_tmp / "sans-corps"
    _release_simulee(d)
    r = _lancer(d, "--sans-flatpak")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "SAUTÉ   corps de release" in r.stdout
