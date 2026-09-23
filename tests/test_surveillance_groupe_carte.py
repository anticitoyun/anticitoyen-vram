"""Pièce 108 : `outils/surveillance-groupe.sh` disait « libre » d'après le seul `.qui`,
qui peut manquer sans que la carte le soit (anticitoyen-vram-jxm : trou entre la mort
réelle d'un service et son .qui ; plus généralement REGLES § 2 : le flock est la seule
vérité, le `.qui` n'est qu'une annonce). Doit désormais interroger flock lui-même quand
le `.qui` est absent, et dire « TENU sans .qui » plutôt que « libre ».
`ACVRAM_VERROU` isole le verrou testé des vrais /tmp/acvram-carte-*.lock du circuit —
jamais touchés ici. `| head -2` coupe la sortie avant le `git fetch` (réseau) du reste
du script : seules les deux premières lignes (date, carte) nous intéressent."""
from __future__ import annotations

import os
import pathlib
import subprocess

RACINE = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = RACINE / "outils" / "surveillance-groupe.sh"


def _ligne_carte(verrou, **env_sup):
    env = {**os.environ, "ACVRAM_VERROU": str(verrou), **env_sup}
    r = subprocess.run(["bash", "-c", f"bash {SCRIPT} | head -2"],
                       cwd=RACINE, env=env, capture_output=True, text=True, timeout=15)
    lignes = r.stdout.splitlines()
    return next((l for l in lignes if l.startswith("carte:")), "")


def test_libre_sans_qui_et_sans_flock(tmp_path):
    verrou = tmp_path / "verrou.lock"   # jamais créé, jamais verrouillé
    assert _ligne_carte(verrou).startswith("carte: libre ")


def test_tenu_avec_qui(tmp_path):
    verrou = tmp_path / "verrou.lock"
    (tmp_path / "verrou.lock.qui").write_text(f"{os.getpid()} 0 une-mesure mesure\n")
    ligne = _ligne_carte(verrou)
    assert ligne.startswith("carte: " + str(os.getpid())), ligne


def test_tenu_sans_qui_dit_tenu_pas_libre(tmp_path):
    """Le cassant : flock tenu par un tiers, AUCUN `.qui` — avant la pièce 108, la
    ligne disait « libre » (mensonge), lu par la seule présence du fichier `.qui`."""
    verrou = tmp_path / "verrou.lock"
    tenant = subprocess.Popen(["flock", str(verrou), "sleep", "10"])
    try:
        ligne = _ligne_carte(verrou)
        assert "TENU sans .qui" in ligne, ligne
        assert "libre" not in ligne, ligne
    finally:
        tenant.terminate()
        tenant.wait(timeout=5)
