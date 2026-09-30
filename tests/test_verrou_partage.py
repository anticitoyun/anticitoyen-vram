"""carte.sh classe PARTAGE (pièce 21) : verrou lecteurs-écrivains à deux fichiers.
PARTAGE (LOCK_SH sur SHARE) coexiste avec d'autres partages ET avec un service ;
une MESURE (LOCK_EX sur SHARE + VERROU) les exclut ; la promesse « ≤ ACVRAM_PROMESSE_MIO,
aucun calcul GPU » est vérifiée par compute-apps → refus + journal, jamais de kill.
Tests à sec, verrou isolé par ACVRAM_VERROU (jamais les vrais /tmp/acvram-carte-*)."""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.usefixtures("recolte_carte")   # ked : rien ne survit au test

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="", **sup)
    return env


def _popen(cmd, env, **sup):
    """Popen dans son PROPRE groupe de processus (start_new_session) : `.terminate()`
    ne signale que le bash de carte.sh, jamais les enfants qu'il a lancés en
    arrière-plan (setsid ou non) — piège trouvé par les orphelins `carte.sh
    sleep N` de ked (27/09). `_finir` signale le GROUPE, pas le seul pid."""
    return subprocess.Popen(cmd, env=env, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **sup)


def _finir(popen):
    """Termine tout le groupe (le bash de carte.sh + tout ce qu'il a lancé),
    jamais le seul pid de tête — sinon un enfant détaché (setsid ou simple `&`
    sans trap) survit, reparenté à init/systemd (ked)."""
    if popen.poll() is not None:
        return
    try:
        os.killpg(popen.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        popen.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(popen.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    popen.wait(timeout=5)


def _tuer_service(srv: str):
    """Un service de carte.sh est détaché par setsid : son PID est chef de groupe ; tuer le groupe."""
    if srv.isdigit():
        try:
            os.killpg(int(srv), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _partage(verrou, secondes, nom):
    """Lance un PARTAGE détaché (sleep) et rend le Popen de carte.sh."""
    return _popen(["bash", str(CARTE), "sleep", str(secondes)],
                  _env(verrou, ACVRAM_TYPE="partage", ACVRAM_NOM=nom))


def test_deux_partages_coexistent(tmp_path):
    verrou = tmp_path / "v.lock"
    a = _partage(verrou, 20, "conv-a")
    b = _partage(verrou, 20, "conv-b")
    time.sleep(2)
    quis = list(tmp_path.glob("v.lock.share.*.qui"))
    try:
        assert len(quis) == 2, f"deux partages → deux .qui, vu {len(quis)}"
    finally:
        _finir(a); _finir(b)


def test_un_partage_ne_bloque_pas_un_service(tmp_path):
    verrou = tmp_path / "v.lock"
    p = _partage(verrou, 20, "conv")
    time.sleep(1.5)
    r = None
    try:
        # un service (LOCK_EX sur VERROU seul) doit démarrer malgré le partage
        r = subprocess.run(["bash", str(CARTE), "sleep", "10"],
                           env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="serveur",
                                    ACVRAM_SERVICE_LOG=str(tmp_path / "s.log")),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, ("service refusé/bloqué par un partage :", r.stderr[-300:])
        srv_pid = r.stdout.strip()
        assert srv_pid.isdigit(), r.stdout
    finally:
        _finir(p)
        # ked : jamais `pkill -f "sleep 10"` (tuait tout « sleep 10 » du poste) — le service est chef de son
        # groupe (setsid de carte.sh) : on tue CE groupe
        _tuer_service(r.stdout.strip() if r else "")


def test_une_mesure_exclut_les_partages(tmp_path):
    verrou = tmp_path / "v.lock"
    p = _partage(verrou, 20, "conv")
    time.sleep(1.5)
    try:
        t0 = time.time()
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="cellule", ACVRAM_ATTENTE="1800"),
                           capture_output=True, text=True, timeout=30)
        dt = time.time() - t0
        assert r.returncode == 4, (r.returncode, r.stderr[-300:])
        assert dt < 10, f"la mesure a attendu ({dt:.1f}s) au lieu d'un refus immédiat"
        assert "REFUS" in r.stderr and "partage" in r.stderr.lower(), r.stderr[-300:]
    finally:
        _finir(p)


def test_deux_mesures_ne_coexistent_pas(tmp_path):
    verrou = tmp_path / "v.lock"
    tenant = _popen(["bash", str(CARTE), "sleep", "20"],
                     _env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="m1"))
    time.sleep(2)
    try:
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="m2", ACVRAM_ATTENTE="2"),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 3, ("2e mesure aurait dû abandonner (attente bornée)", r.returncode, r.stderr[-200:])
    finally:
        _finir(tenant)


def test_partage_refuse_pendant_mesure(tmp_path):
    """Un partagé qui arrive pendant une MESURE est refusé (LOCK_SH sur S bloqué
    par le LOCK_EX de la mesure)."""
    verrou = tmp_path / "v.lock"
    m = _popen(["bash", str(CARTE), "sleep", "20"],
               _env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="cellule"))
    time.sleep(2)
    try:
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="partage", ACVRAM_NOM="conv"),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 4, (r.returncode, r.stderr[-300:])
        assert "REFUS" in r.stderr, r.stderr[-300:]
    finally:
        _finir(m)


def test_partage_admis_pendant_service(tmp_path):
    """Un partagé qui arrive pendant un SERVICE est admis (fichiers disjoints :
    service sur P, partagé sur S)."""
    verrou = tmp_path / "v.lock"
    r = subprocess.run(["bash", str(CARTE), "sleep", "15"],
                       env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="serveur",
                                ACVRAM_SERVICE_LOG=str(tmp_path / "s.log")),
                       capture_output=True, text=True, timeout=30)
    srv = r.stdout.strip()
    p = _partage(verrou, 8, "conv")
    time.sleep(2)
    try:
        quis = list(tmp_path.glob("v.lock.share.*.qui"))
        assert quis, "le partagé n'a pas été admis pendant le service"
        assert p.poll() is None, "le partagé s'est arrêté (aurait dû tourner)"
    finally:
        _finir(p)
        _tuer_service(srv)


def test_promesse_violee_refuse_sans_kill(tmp_path):
    """Un partage qui trahit sa promesse (> seuil sur le GPU) fait refuser la
    prise SUIVANTE (un 2e partage, qui coexiste et vérifie les promesses) +
    PROMESSE-VIOLEE au journal, sans tuer le fautif. compute-apps simulé par un
    faux nvidia-smi en tête de PATH (à sec, pas de vrai GPU)."""
    verrou = tmp_path / "v.lock"
    faux = tmp_path / "bin"; faux.mkdir()
    # faux nvidia-smi : chaque partage déclaré alloue 999 Mio (viole la promesse).
    (faux / "nvidia-smi").write_text(
        "#!/bin/bash\n"
        f'for q in "{verrou}".share.*.qui; do [ -e "$q" ] || continue; read -r p _ < "$q"; echo "$p, 999"; done\n')
    (faux / "nvidia-smi").chmod(0o755)
    p = _partage(verrou, 20, "gourmand")
    time.sleep(1.5)
    try:
        env = _env(verrou, ACVRAM_TYPE="partage", ACVRAM_NOM="second", ACVRAM_PROMESSE_MIO="512")
        env["PATH"] = f"{faux}:{env['PATH']}"
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"], env=env,
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 5, (r.returncode, r.stderr[-300:])
        assert "PROMESSE" in r.stderr.upper(), r.stderr[-300:]
        journal = (tmp_path / "v.lock.journal").read_text()
        assert "PROMESSE-VIOLEE" in journal, journal
        assert p.poll() is None, "le partage fautif a été tué (interdit)"
    finally:
        _finir(p)


def test_finir_ne_laisse_aucun_orphelin(tmp_path):
    """ked (27/09) : `.terminate()` ne signale que le bash de carte.sh, jamais
    le `sleep` qu'il a lancé en arrière-plan (classe PARTAGE, pas de setsid ni
    de trap TERM) — orphelin reparenté à init, `carte.sh sleep N` en file
    après la suite. `_finir` doit couper tout le groupe."""
    verrou = tmp_path / "v.lock"
    p = _partage(verrou, 20, "conv")
    time.sleep(1.5)
    pgid = os.getpgid(p.pid)
    _finir(p)
    time.sleep(0.5)
    survivants = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
    assert not survivants, f"processus survivants dans le groupe {pgid} : {survivants}"
