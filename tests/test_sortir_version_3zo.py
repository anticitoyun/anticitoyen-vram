"""Pièce 3zo (chef, 29/09, sortie 0.7.13) : l'étape 7 de `outils/sortir-version.sh` lançait
`verifier-release.sh` hors carte.sh ; le shell du chef porte CUDA_VISIBLE_DEVICES="" (à sec), le
doctor Flatpak ne voyait aucun GPU et le script concluait « FAUX Marlin non chargé » (rc 71) sur une
release bonne — rejouée à la main sous carte.sh : TENU. Correctif : (a) l'étape 7 scinde le bras
installation (hors carte) et le bras doctor (sous outils/carte.sh, qui rétablit CUDA_VISIBLE_DEVICES) ;
(b) verifier-release.sh REFUSE (67) quand CUDA_VISIBLE_DEVICES est défini vide et que le doctor doit
tourner, au lieu de conclure sur l'absence de Marlin. Dépôt jetable (281), carte.sh et
verifier-release.sh remplacés par des témoins qui journalisent — aucune prise de carte réelle."""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_sortir_version_281 import _depot_jetable, V  # noqa: E402

RACINE = pathlib.Path(__file__).resolve().parents[1]
VERIFIE = RACINE / "outils" / "verifier-release.sh"


def _lancer(depot: pathlib.Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(depot / "outils" / "sortir-version.sh"), *args],
                          cwd=depot, capture_output=True, text=True, timeout=60, env=dict(os.environ, **env))


def _temoins(depot: pathlib.Path) -> pathlib.Path:
    """carte.sh et verifier-release.sh témoins : chacun note ses arguments et l'état de
    CUDA_VISIBLE_DEVICES dans un journal ; carte.sh rétablit la carte comme le vrai."""
    journal = depot / "journal.txt"
    outils = depot / "outils"
    (outils / "carte.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"carte.sh $*\" >> {journal}\n"
        "exec env CUDA_VISIBLE_DEVICES=0 \"$@\"\n", encoding="utf-8")
    (outils / "verifier-release.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"verifier-release.sh $* CVD=${{CUDA_VISIBLE_DEVICES-absent}}\" >> {journal}\n"
        "exit 0\n", encoding="utf-8")
    for f in ("carte.sh", "verifier-release.sh"):
        (outils / f).chmod(0o755)
    return journal


# ---- (a) sortir-version : le doctor passe sous carte.sh -------------------------------------------

def test_simule_montre_le_doctor_sous_carte_sh(tmp_path):
    d = _depot_jetable(tmp_path)
    r = _lancer(d, {}, f"v{V}", "--simule")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"[simulé] outils/verifier-release.sh v{V} --flatpak-installer" in r.stdout, r.stdout
    assert f"[simulé] outils/carte.sh outils/verifier-release.sh v{V} --flatpak-doctor" in r.stdout, r.stdout


def test_etape_7_reelle_lance_le_doctor_sous_carte_sh_et_l_installation_hors_carte(tmp_path):
    """--depuis 7 sur un tag posé : rien n'est poussé, seule l'étape 7 tourne — avec le shell
    à sec du chef (CUDA_VISIBLE_DEVICES=""), le bras doctor doit voir la carte rétablie par
    carte.sh, le bras installation reste hors carte."""
    d = _depot_jetable(tmp_path)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "v"], check=True)
    journal = _temoins(d)
    r = _lancer(d, {"CUDA_VISIBLE_DEVICES": ""}, f"v{V}", "--depuis", "7")
    assert r.returncode == 0, r.stdout + r.stderr
    lignes = journal.read_text(encoding="utf-8").splitlines()
    assert lignes == [
        f"verifier-release.sh v{V} --flatpak-installer CVD=",
        f"carte.sh outils/verifier-release.sh v{V} --flatpak-doctor",
        f"verifier-release.sh v{V} --flatpak-doctor CVD=0",
    ], lignes


def test_un_defaut_du_bras_doctor_refuse_71(tmp_path):
    d = _depot_jetable(tmp_path)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "v"], check=True)
    _temoins(d)
    (d / "outils" / "verifier-release.sh").write_text(
        '#!/usr/bin/env bash\ncase "$*" in *--flatpak-doctor*) exit 1 ;; esac\nexit 0\n', encoding="utf-8")
    r = _lancer(d, {}, f"v{V}", "--depuis", "7")
    assert r.returncode == 71 and "bras doctor sous carte.sh" in r.stderr, r.stderr


# ---- (b) verifier-release : refus net à sec, jamais « Marlin absent » ---------------------------

@pytest.fixture
def hors_tmp():
    """verifier-release.sh refuse /tmp (259) : dossier simulé sous le cache, nettoyé après."""
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "acvram" / "tests-3zo"
    d = base / f"p{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _verifier(hors_tmp: pathlib.Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(VERIFIE), f"v{V}", "--simule", str(hors_tmp), *args],
                          capture_output=True, text=True, timeout=60, env=dict(os.environ, **env))


@pytest.mark.parametrize("bras", [(), ("--flatpak-doctor",)])
def test_refuse_67_a_sec_quand_le_doctor_doit_tourner(hors_tmp, bras):
    r = _verifier(hors_tmp, {"CUDA_VISIBLE_DEVICES": ""}, *bras)
    assert r.returncode == 67, r.stdout + r.stderr
    assert "CUDA_VISIBLE_DEVICES est défini vide" in r.stderr and "carte.sh" in r.stderr, r.stderr
    assert "Marlin" not in r.stdout, r.stdout


@pytest.mark.parametrize("bras", [("--sans-flatpak",), ("--flatpak-installer",)])
def test_a_sec_sans_doctor_ne_refuse_pas(hors_tmp, bras):
    """Le refus ne vise que le bras qui a besoin de la carte : installation et --sans-flatpak
    passent à sec (le dossier simulé est vide : verdict FAUX ordinaire, code 1, pas 67)."""
    r = _verifier(hors_tmp, {"CUDA_VISIBLE_DEVICES": ""}, *bras)
    assert r.returncode == 1 and "REFUS" not in r.stderr, r.stdout + r.stderr


def test_carte_visible_ne_refuse_pas(hors_tmp):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    r = subprocess.run(["bash", str(VERIFIE), f"v{V}", "--simule", str(hors_tmp), "--flatpak-doctor"],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 1 and "REFUS" not in r.stderr, r.stdout + r.stderr
