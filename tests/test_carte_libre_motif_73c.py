"""Pièce 73c : carte-libre.sh ne doit pas prendre le crochet masque-jetons.py
(~/.config/acvram/) pour une mesure — seule coïncidence : le mot « acvram »
dans son chemin. Test cassant : rétabli à l'ancien motif, il doit échouer.

Hermétique (chef, 27/09, après échec sous carte réellement occupée) : un
faux nvidia-smi en tête de PATH répond toujours « 0 Mio, aucune app » — le
verdict ne dépend plus de ce qui tourne sur la machine au moment du test.
"""
import os
import signal
import stat
import subprocess
import sys
import time

DEPOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(DEPOT, "outils", "carte-libre.sh")

FAUX_NVIDIA_SMI = """#!/bin/sh
# nvidia-smi simule : carte toujours vide, quelle que soit la question posee.
case "$*" in
  *query-gpu=memory.used*) echo 0 ;;
  *) exit 0 ;;
esac
"""


def _faux_path(tmp_path):
    bindir = tmp_path / "faux-bin"
    bindir.mkdir()
    cible = bindir / "nvidia-smi"
    cible.write_text(FAUX_NVIDIA_SMI)
    cible.chmod(cible.stat().st_mode | stat.S_IEXEC)
    return str(bindir) + os.pathsep + os.environ["PATH"]


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


def test_crochet_masque_jetons_nest_pas_une_mesure(tmp_path):
    proc, fichier = _lance_faux_processus(
        os.path.join("scratchpad-73c", ".config", "acvram", "masque-jetons.py")
    )
    try:
        r = subprocess.run(
            [SCRIPT],
            cwd=DEPOT,
            env={**os.environ, "RACINE": "1", "PATH": _faux_path(tmp_path)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "masque-jetons" not in r.stderr, r.stderr
    finally:
        _nettoie(proc, fichier)


def test_vrai_outils_reste_attrape_comme_intrus(tmp_path):
    proc, fichier = _lance_faux_processus(
        os.path.join("outils", "faux-mesure-73c.py")
    )
    try:
        r = subprocess.run(
            [SCRIPT],
            cwd=DEPOT,
            env={**os.environ, "RACINE": "1", "PATH": _faux_path(tmp_path)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 1
        assert "une mesure demarre sans avoir encore alloue" in r.stderr, r.stderr
    finally:
        _nettoie(proc, fichier)
