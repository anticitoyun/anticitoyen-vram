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


def test_gardien_n_efface_pas_le_qui_d_un_nouveau_detenteur(tmp_path):
    """anticitoyen-vram-jxm : le gardien d'un service NE POLLE qu'toutes les 5 s.
    Si, entre la mort du service et le prochain reveil du gardien, un NOUVEAU
    detenteur a deja pris le verrou libere et ecrit SON `.qui` a la meme adresse
    (fichier partage par carte), le gardien ne doit PAS l'effacer : il ne connait
    QUE son propre PID de service, mort depuis. Reproduit sans attendre le vrai
    delai de 5 s : le faux service meurt vite (sleep 1), on remplace `.qui` par
    un nouveau detenteur AUSSITOT apres sa mort (avant le premier reveil du
    gardien, qui n'a pas encore eu lieu), puis on attend le prochain reveil pour
    verifier qu'il n'a rien touche."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="service-court",
                                ACVRAM_SERVICE_LOG=str(tmp_path / "srv.log")),
                       capture_output=True, text=True, timeout=30)
    ancien_pid = r.stdout.strip()
    # attendre la mort REELLE du service (~1 s), avant le premier reveil du gardien (5 s)
    deadline = time.time() + 4
    while time.time() < deadline and subprocess.run(["kill", "-0", ancien_pid]).returncode == 0:
        time.sleep(0.1)
    assert subprocess.run(["kill", "-0", ancien_pid]).returncode != 0, "le faux service (sleep 1) devrait être mort"
    # un NOUVEAU detenteur prend la place, tout de suite : son .qui a un AUTRE pid
    nouveau = subprocess.Popen(["sleep", "20"])
    try:
        info.write_text(f"{nouveau.pid} {int(time.time())} nouveau-detenteur mesure\n")
        # laisser le temps au gardien de se reveiller au moins une fois (poll 5 s)
        # et de constater la mort de L'ANCIEN pid — sans toucher au .qui du nouveau
        time.sleep(7)
        assert info.exists(), ".qui du NOUVEAU détenteur effacé par le gardien de l'ANCIEN service"
        champs = info.read_text().split()
        assert champs[0] == str(nouveau.pid), f".qui altéré : {champs}"
    finally:
        nouveau.terminate()
        nouveau.wait(timeout=5)


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


def test_deux_services_partagent_la_carte_et_une_mesure_reste_refusee(tmp_path):
    """27/09 (bd kmb, ComfyUI + llama-server d'Open WebUI sur la 5090) : un second
    service ne fait PAS la queue derriere le premier (verrou partage), le `.qui`
    porte une ligne par service vivant, une mesure reste refusee (4) tant qu'un
    service vit, et la mort du second ne retire que SA ligne. Sans le partage,
    le second carte.sh attendrait ATTENTE puis abandonnerait (3) : cassant."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    a = subprocess.run(["bash", str(CARTE), "sleep", "30"],
                       env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="llama-owui",
                                ACVRAM_SERVICE_LOG=str(tmp_path / "a.log")),
                       capture_output=True, text=True, timeout=30)
    pid_a = a.stdout.strip()
    try:
        t0 = time.time()
        b = subprocess.run(["bash", str(CARTE), "sleep", "2"],
                           env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="comfyui",
                                    ACVRAM_SERVICE_LOG=str(tmp_path / "b.log"), ACVRAM_ATTENTE="6"),
                           capture_output=True, text=True, timeout=30)
        dt = time.time() - t0
        assert b.returncode == 0 and dt < 5, (b.returncode, dt, b.stderr[-300:])
        pid_b = b.stdout.strip()
        lignes = info.read_text().splitlines()
        assert [l.split()[0] for l in lignes] == [pid_a, pid_b], lignes
        assert all(l.split()[3] == "service" for l in lignes), lignes
        assert not _flock_libre(verrou), "verrou exclusif libre alors que deux services vivent"
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure"),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 4 and "llama-owui" in r.stderr and "comfyui" in r.stderr, (r.returncode, r.stderr[-300:])
        # le second meurt (sleep 2) : son gardien (poll 5 s) retire SA ligne seulement
        for _ in range(120):
            if info.exists() and [l.split()[0] for l in info.read_text().splitlines()] == [pid_a]:
                break
            time.sleep(0.1)
        assert [l.split()[0] for l in info.read_text().splitlines()] == [pid_a], info.read_text()
        assert not _flock_libre(verrou), "verrou libre alors que le premier service vit encore"
    finally:
        subprocess.run(["kill", pid_a], check=False)
    for _ in range(120):
        if not info.exists() and _flock_libre(verrou):
            break
        time.sleep(0.1)
    assert not info.exists() and _flock_libre(verrou)


def test_mesure_sans_qui_attend_tant_qu_un_service_tient_le_verrou_partage(tmp_path):
    """Garantie demandée par chef (27/09) : une MESURE prend le verrou en EXCLUSIF,
    donc elle ne passe JAMAIS pendant qu'un service tient le verrou partagé — même
    si le `.qui` a disparu (cas jxm) : elle attend, puis abandonne (3) à ATTENTE,
    au lieu de mesurer à côté du service. Cassant si un service prenait le
    verrou autrement qu'en flock (la mesure passerait, code 0)."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    svc = subprocess.run(["bash", str(CARTE), "sleep", "30"],
                         env=_env(verrou, ACVRAM_TYPE="service", ACVRAM_NOM="serveur",
                                  ACVRAM_SERVICE_LOG=str(tmp_path / "srv.log")),
                         capture_output=True, text=True, timeout=30)
    pid = svc.stdout.strip()
    try:
        info.unlink()                       # .qui perdu : le flock seul doit protéger
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure",
                                    ACVRAM_ATTENTE="3", ACVRAM_TICKET_DESACTIVE="1"),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 3 and "ABANDON" in r.stderr, (r.returncode, r.stderr[-300:])
    finally:
        subprocess.run(["kill", pid], check=False)
