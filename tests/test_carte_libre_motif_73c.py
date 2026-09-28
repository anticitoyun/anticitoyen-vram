"""Pièce 73c : carte-libre.sh ne doit pas prendre le crochet masque-jetons.py
(~/.config/acvram/) pour une mesure — seule coïncidence : le mot « acvram »
dans son chemin. Test cassant : rétabli à l'ancien motif, il doit échouer.

Hermétique (chef, 27/09, après échec sous carte réellement occupée) : un
faux nvidia-smi en tête de PATH répond toujours « 0 Mio, aucune app » — le
verdict ne dépend plus de ce qui tourne sur la machine au moment du test.

g2c (28/09, ordre chef) : depuis g2c, un candidat MOTIF qui ne tient NI un
`.qui` vivant NI de mémoire GPU réelle n'est plus qu'un avertissement (une
campagne EN PAUSE ne bloque plus personne) — ce que `test_vrai_outils_...`
simulait (processus lancé, « 0 Mio, aucune app » partout) n'est donc plus un
« vrai intrus » AUJOURD'HUI. Un vrai intrus est un processus qui tient
réellement de la mémoire GPU (ou un `.qui`) : le faux nvidia-smi de ce test
le déclare désormais dans `--query-compute-apps`, ce qui le fait attraper
par le critère 2 (mémoire non déclarée), message « Intrus » — et non plus
par le motif d'intentions (critère 3, « une mesure demarre... », réservé
maintenant à une vraie prise carte.sh en train de démarrer). g2c n'est pas
affaibli par ce changement : c'est ce test qui décrivait un intrus à côté de
la plaque."""
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


def _faux_path_avec_gpu(tmp_path, pid, mem_mio=512):
    """Même faux nvidia-smi, mais ce PID précis tient réellement mem_mio de
    mémoire GPU (compute-apps) — ce qui fait un intrus aujourd'hui (g2c)."""
    bindir = tmp_path / "faux-bin-gpu"
    bindir.mkdir()
    cible = bindir / "nvidia-smi"
    cible.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f'  *query-gpu=memory.used*) echo {mem_mio} ;;\n'
        f'  *query-compute-apps=pid,used_memory*) echo "{pid}, {mem_mio}" ;;\n'
        f'  *query-compute-apps=pid*) echo "{pid}" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
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
            env={**os.environ, "RACINE": "1", "PATH": _faux_path_avec_gpu(tmp_path, proc.pid)},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert r.returncode == 1
        assert "Intrus" in r.stderr, r.stderr
        assert str(proc.pid) in r.stderr, r.stderr
    finally:
        _nettoie(proc, fichier)
