"""anticitoyen-vram-g2c (28/09) : interblocage 12h05-13h47 — carte-libre.sh
comptait la campagne 290 (poste2) EN PAUSE (`campagne-tps-menus.py`, qui
attend un drapeau `~/.cache/acvram/campagne290/pause`, sans prise carte.sh
ni mémoire GPU) comme « une mesure démarre sans avoir encore alloué » :
poste1 attendait carte-libre, la campagne attendait le drapeau d'poste1.

Un candidat qui matche MOTIF (pgrep) mais ne tient NI un `.qui` vivant de
carte.sh NI de mémoire GPU réelle (nvidia-smi compute-apps) n'est qu'un
avertissement — jamais un refus. La protection reste entière pour une VRAIE
prise qui vient de démarrer (verrou pris, allocation pas encore visible),
sinon ce correctif rouvrirait exactement le trou que ce refus visait.

`nvidia-smi` factice (PATH, aucune allocation nulle part) ; le `.qui` est
soit absent (cas g2c), soit écrit avec le pid réel (cas protégé). Aucune
carte réelle requise.

bd cn1 (chef, 01/10) : les deux tests ci-dessus échouaient dans la suite complète pendant
qu'un vrai poste compilait des noyaux (nvcc/ninja, typique d'un chargement de modèle) — le
troisième pgrep de `carte-libre.sh` (détection de compilation, `exit 2`) n'était isolé par
AUCUNE variable, contrairement à `MOTIF`. `MOTIF_COMPIL` (paramétrée ci-dessous) corrige ça ;
`test_isolation_compil_casse_sans_motif_compil` est le contrôle demandé : un leurre nommé
« ninja » tourne PENDANT les deux tests existants (prouvant qu'ils resteraient verts même si
MOTIF_COMPIL fuit n'était pas le bon réglage), et ce contrôle prouve DIRECTEMENT que ce même
leurre, sans l'isolation, fait bien échouer carte-libre.sh (rc 2) — si quelqu'un retire
MOTIF_COMPIL des env de ce fichier, ce contrôle continue de passer mais
`test_pause_sans_verrou_ni_gpu_nest_plus_un_refus` échouerait alors (rc 2 au lieu de 0)."""
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

DEPOT = Path(__file__).resolve().parent.parent
SCRIPT = DEPOT / "outils" / "carte-libre.sh"

FAUX_NVIDIA_SMI = """#!/bin/sh
case "$*" in
  *query-gpu=memory.used*) echo 0 ;;
  *pmon*) exit 0 ;;
  *query-compute-apps*) exit 0 ;;
  *) exit 0 ;;
esac
"""


def _faux_path(tmp_path):
    d = tmp_path / "faux-bin"
    d.mkdir(exist_ok=True)
    f = d / "nvidia-smi"
    f.write_text(FAUX_NVIDIA_SMI)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(d) + os.pathsep + os.environ["PATH"]


def _lancer_script_en_pause(depot_leurre):
    """Un « outils/campagne-…-menus.py » qui dort — matche MOTIF
    (python.*(outils/|acvram)), ne prend jamais carte.sh, n'alloue jamais de
    GPU. Le procédé exact de la 290 : en pause, hors prise."""
    dossier = depot_leurre / "outils"
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = dossier / "campagne-tps-menus.py"
    fichier.write_text("import time\ntime.sleep(30)\n")
    proc = subprocess.Popen(["python3", str(fichier)], cwd=str(depot_leurre))
    time.sleep(0.3)
    return proc, fichier


def _nettoyer(proc, fichier):
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=5)
    try:
        fichier.unlink()
    except OSError:
        pass


# bd cn1 : décor — un leurre dont le NOM DU BINAIRE est « ninja », le motif par défaut de
# MOTIF_COMPIL (ancré sur `(^|/)(nvcc|cicc|ptxas|cudafe\+\+|ninja)( |$)`). Tourne pendant les
# deux tests ci-dessous : si MOTIF_COMPIL cesse d'être isolé (retiré de leur env), ce leurre
# les fait échouer immédiatement — c'est le contrôle de fuite demandé par chef.
def _lancer_leurre_compilation(tmp_path):
    faux_ninja = tmp_path / "ninja"
    faux_ninja.write_text("#!/bin/sh\nsleep 30\n")
    faux_ninja.chmod(faux_ninja.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.Popen([str(faux_ninja)])
    time.sleep(0.2)
    return proc


def test_pause_sans_verrou_ni_gpu_nest_plus_un_refus(tmp_path):
    proc, fichier = _lancer_script_en_pause(tmp_path)
    leurre = _lancer_leurre_compilation(tmp_path)
    try:
        r = subprocess.run(
            [str(SCRIPT)], cwd=DEPOT,
            env={**os.environ, "RACINE": "1", "PATH": _faux_path(tmp_path),
                 "QUI_CARTE": str(tmp_path / "aucun.qui"),
                 "MOTIF_COMPIL": "__jamais_aucune_correspondance__"},
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, r.stderr
        assert "avertissement, pas un refus" in r.stderr, r.stderr
        assert "une mesure demarre sans avoir encore alloue" not in r.stderr, r.stderr
    finally:
        leurre.send_signal(signal.SIGKILL); leurre.wait(timeout=5)
        _nettoyer(proc, fichier)


def test_isolation_compil_casse_sans_motif_compil(tmp_path):
    """Contrôle demandé (chef, bd cn1) : le MÊME leurre « ninja » que ci-dessus, sans
    l'isolation MOTIF_COMPIL — prouve que le leurre matche bien le motif réel et ferait
    échouer `test_pause_sans_verrou_ni_gpu_nest_plus_un_refus` si l'isolation disparaissait."""
    proc, fichier = _lancer_script_en_pause(tmp_path)
    leurre = _lancer_leurre_compilation(tmp_path)
    try:
        r = subprocess.run(
            [str(SCRIPT)], cwd=DEPOT,
            env={**os.environ, "RACINE": "1", "PATH": _faux_path(tmp_path),
                 "QUI_CARTE": str(tmp_path / "aucun.qui")},   # pas de MOTIF_COMPIL : non isolé
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 2, (r.returncode, r.stderr)
        assert "processus de compilation en cours" in r.stderr, r.stderr
    finally:
        leurre.send_signal(signal.SIGKILL); leurre.wait(timeout=5)
        _nettoyer(proc, fichier)


def test_vraie_mesure_sous_verrou_reste_refusee(tmp_path):
    """Contrôle : un candidat qui tient un `.qui` vivant (verrou pris, rien
    encore alloué au GPU) reste refusé — la protection d'origine n'a pas
    disparu, seul le faux positif de la pause a été retiré."""
    proc, fichier = _lancer_script_en_pause(tmp_path)
    try:
        qui = tmp_path / "carte-0.lock.qui"
        qui.write_text(f"{proc.pid} 1234567890 campagne290 mesure {proc.pid}\n")
        r = subprocess.run(
            [str(SCRIPT)], cwd=DEPOT,
            env={**os.environ, "RACINE": "1", "PATH": _faux_path(tmp_path),
                 "QUI_CARTE": str(qui)},
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 1, (r.returncode, r.stderr)
        assert "une mesure demarre sans avoir encore alloue" in r.stderr, r.stderr
    finally:
        _nettoyer(proc, fichier)
