"""Pièce 73c : carte-libre.sh ne doit pas prendre le crochet masque-jetons.py
(~/.config/acvram/) pour une mesure — seule coïncidence : le mot « acvram »
dans son chemin. Test cassant : rétabli à l'ancien motif, il doit échouer."""
import os
import signal
import subprocess
import sys
import time

DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(DEPOT, "outils", "carte-libre.sh")


def _lance_faux_processus(chemin_relatif):
    dossier = os.path.dirname(os.path.join(DEPOT, chemin_relatif))
    os.makedirs(dossier, exist_ok=True)
    fichier = os.path.join(DEPOT, chemin_relatif)
    with open(fichier, "w") as f:
        f.write("import time\ntime.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, fichier])
    time.sleep(0.3)
    return proc, fichier


def _nettoie(proc, fichier):
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    try:
        os.remove(fichier)
    except OSError:
        pass


def test_crochet_masque_jetons_nest_pas_une_mesure():
    proc, fichier = _lance_faux_processus(
        os.path.join("scratchpad-73c", ".config", "acvram", "masque-jetons.py")
    )
    try:
        r = subprocess.run(
            [SCRIPT],
            cwd=DEPOT,
            env={**os.environ, "RACINE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "masque-jetons" not in r.stderr, r.stderr
    finally:
        _nettoie(proc, fichier)


def test_vrai_outils_reste_attrape_comme_intrus():
    proc, fichier = _lance_faux_processus(
        os.path.join("outils", "faux-mesure-73c.py")
    )
    try:
        r = subprocess.run(
            [SCRIPT],
            cwd=DEPOT,
            env={**os.environ, "RACINE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 1
        assert "une mesure demarre sans avoir encore alloue" in r.stderr, r.stderr
    finally:
        _nettoie(proc, fichier)
