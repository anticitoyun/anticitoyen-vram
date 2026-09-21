"""carte.sh ACVRAM_TYPE=service (trou du 21/09 : acvram/llamacpp/vllm-serveur
allouaient la carte en `setsid nohup` sans verrou → une mesure la croyait libre,
REGLES § 2). Mode service : le serveur DETACHE herite le descripteur du flock et
tient donc le verrou lui-meme ; carte.sh sort aussitot (rend la main au lanceur),
le `.qui` apparait avec le PID du serveur, et un gardien l'efface a sa mort — le
verrou, lui, tombe des que le serveur meurt. Refus mutuel nomme service<->mesure
(code 4, sans attendre). Tests a sec avec un faux serveur (`sleep`), verrou isole
par ACVRAM_VERROU : jamais les vrais /tmp/acvram-carte-*.lock du circuit."""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="", **sup)
    return env


def _flock_libre(verrou):
    return subprocess.run(["flock", "-n", str(verrou), "true"]).returncode == 0


def test_service_detache_pose_le_qui_et_rend_la_main(tmp_path):
    """carte.sh service revient AUSSITOT (il ne wait pas le serveur), imprime le
    PID du serveur, pose un `.qui` `<pid> <epoch> <nom> service`, et le verrou est
    tenu par le serveur detache (un flock non bloquant echoue)."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    t0 = time.time()
    r = subprocess.run(["bash", str(CARTE), "sleep", "20"],
                       env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="faux-serveur",
                                ACVRAM_SERVICE_LOG=str(tmp_path / "srv.log")),
                       capture_output=True, text=True, timeout=30)
    dt = time.time() - t0
    assert r.returncode == 0, r.stderr[-300:]
    assert dt < 10, f"carte.sh service a attendu le serveur ({dt:.1f}s) au lieu de detacher"
    pid = r.stdout.strip()
    assert pid.isdigit(), f"stdout n'est pas le PID du serveur : {r.stdout!r}"
    # le .qui est apparu, au format 4 champs, avec le PID du serveur et le type service
    champs = info.read_text().split()
    assert champs[0] == pid and champs[2] == "faux-serveur" and champs[3] == "service", champs
    # le verrou est REELLEMENT tenu par le serveur detache
    assert not _flock_libre(verrou), "verrou libre alors que le service tourne"
    # nettoyage : tuer le faux serveur
    subprocess.run(["kill", pid], check=False)


def test_le_qui_disparait_et_le_verrou_tombe_a_la_mort_du_serveur(tmp_path):
    """A la mort du serveur : le verrou tombe (fd libere) et le gardien efface le
    `.qui`. C'est le cassant — sans le mode service (repli `setsid nohup`), aucun
    `.qui` n'apparaitrait et le verrou ne serait jamais pris."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    r = subprocess.run(["bash", str(CARTE), "sleep", "2"],
                       env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="ephemere",
                                ACVRAM_SERVICE_LOG=str(tmp_path / "srv.log")),
                       capture_output=True, text=True, timeout=30)
    pid = r.stdout.strip()
    assert info.exists(), "le .qui n'est pas apparu"
    assert not _flock_libre(verrou), "verrou libre pendant que le service vit"
    # le serveur (sleep 2) meurt seul ; le gardien verifie toutes les 5 s
    deadline = time.time() + 20
    while time.time() < deadline and (info.exists() or not _flock_libre(verrou)):
        time.sleep(1)
    assert not info.exists(), ".qui toujours present apres la mort du serveur"
    assert _flock_libre(verrou), "verrou toujours tenu apres la mort du serveur"


def test_mesure_refusee_nommee_pendant_qu_un_service_tient(tmp_path):
    """Une prise de mesure arrivant pendant qu'un service tient est REFUSEE tout
    de suite (code 4), nommant le detenteur, sans entrer dans l'attente de 30 min."""
    verrou = tmp_path / "verrou.lock"
    svc = subprocess.run(["bash", str(CARTE), "sleep", "20"],
                        env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="serveur-en-place",
                                 ACVRAM_SERVICE_LOG=str(tmp_path / "srv.log")),
                        capture_output=True, text=True, timeout=30)
    pid = svc.stdout.strip()
    try:
        t0 = time.time()
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                          env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure",
                                   ACVRAM_ATTENTE="1800"),
                          capture_output=True, text=True, timeout=30)
        dt = time.time() - t0
        assert r.returncode == 4, (r.returncode, r.stderr[-300:])
        assert dt < 10, f"la mesure a attendu ({dt:.1f}s) au lieu d'un refus immediat"
        assert "REFUS" in r.stderr and "serveur-en-place" in r.stderr, r.stderr[-300:]
    finally:
        subprocess.run(["kill", pid], check=False)


def test_service_refuse_nomme_pendant_qu_une_mesure_tient(tmp_path):
    """L'inverse : un service demande pendant qu'une mesure tient est refuse (4)."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    # une mesure qui tient le verrou : un flock detenu par un `sleep` que NOUS
    # gardons vivant, avec un .qui de type mesure a PID vivant.
    tenant = subprocess.Popen(["flock", str(verrou), "sleep", "20"])
    time.sleep(0.5)
    info.write_text(f"{tenant.pid} {int(time.time())} une-mesure mesure\n")
    try:
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                          env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="un-service"),
                          capture_output=True, text=True, timeout=30)
        assert r.returncode == 4, (r.returncode, r.stderr[-300:])
        assert "REFUS" in r.stderr, r.stderr[-300:]
    finally:
        tenant.terminate()
