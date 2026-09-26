"""Pièce 244 (chef, 26/09) : un `kill` externe sur carte.sh LUI-MÊME (pas son propre
garde DUREE_MAX) le tuait sans jamais toucher la commande — encore vivante, hors verrou
(3e contamination en trois jours, poste6 03:13, un pytest orphelin 10 min pendant la
prise d'poste1). Deux correctifs testés ici, à sec, verrou isolé (ACVRAM_VERROU) :
(1) la commande tourne dans son propre groupe de processus (`setsid`) et un TERM/INT/HUP
reçu par carte.sh est transmis à CE groupe avant de rendre le verrou ;
(2) à la prise suivante, un `.qui` dont le 5e champ (pgid) désigne un groupe encore vivant
refuse la carte plutôt que de la rendre en silence par-dessus une commande qui tourne
encore (cas KILL -9, qu'aucun trap ne peut rattraper)."""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import time

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="", **sup)
    return env


def _vivant(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_sigterm_du_carte_sh_tue_tout_le_groupe_de_la_commande(tmp_path):
    """Le cas réel du 26/09 : la commande lance elle-même un enfant (un pytest lancerait
    des sous-processus) ; un TERM sur carte.sh doit achever l'enfant ET le petit-enfant."""
    verrou = tmp_path / "verrou.lock"
    pidfile = tmp_path / "pids"
    cmd = f'echo $$ > {pidfile}; sleep 120 & echo $! >> {pidfile}; wait'
    proc = subprocess.Popen(["bash", str(CARTE), "bash", "-c", cmd],
                            env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="m244", ACVRAM_DUREE_MAX="0"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 10
    while time.time() < deadline and not (pidfile.exists() and len(pidfile.read_text().split()) == 2):
        time.sleep(0.05)
    assert pidfile.exists(), "la commande n'a jamais écrit ses PID"
    pere, enfant = (int(x) for x in pidfile.read_text().split())
    assert _vivant(pere) and _vivant(enfant), "la commande ou son enfant n'a jamais démarré"

    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)

    assert not _vivant(pere), "le père (commande directe de carte.sh) a survécu au TERM"
    assert not _vivant(enfant), "le petit-enfant (lancé PAR la commande) a survécu au TERM — c'est le bug du 26/09"


def test_qui_avec_pgid_vivant_refuse_la_carte(tmp_path):
    """Un `.qui` (5 champs) dont le pgid désigne un groupe encore vivant — cas KILL -9,
    qu'aucun trap ne rattrape — doit REFUSER la prise suivante, pas la rendre en silence."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    groupe = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not _vivant(groupe.pid):
            time.sleep(0.02)
        info.write_text(f"999999 0 ancien-detenteur mesure {groupe.pid}\n")
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="m244b", ACVRAM_DUREE_MAX="0"),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 4, r.stdout + r.stderr
        assert "REFUS" in r.stderr and str(groupe.pid) in r.stderr, r.stderr
    finally:
        groupe.terminate()
        groupe.wait(timeout=5)


def test_qui_avec_pgid_mort_n_empeche_pas_la_prise(tmp_path):
    """Contrôle : un `.qui` pointant un groupe déjà mort ne doit RIEN refuser (sinon le
    contrôle refuserait toujours, capable de rendre faux dans l'autre sens)."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    groupe = subprocess.Popen(["sleep", "1"], start_new_session=True)
    pgid_mort = groupe.pid
    groupe.wait(timeout=5)
    deadline = time.time() + 5
    while time.time() < deadline and _vivant(pgid_mort):
        time.sleep(0.05)
    assert not _vivant(pgid_mort), "le groupe témoin n'est jamais mort"
    info.write_text(f"999999 0 ancien-detenteur mesure {pgid_mort}\n")
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="m244c", ACVRAM_DUREE_MAX="0"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
