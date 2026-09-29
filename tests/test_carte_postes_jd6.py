"""jd6 (29/09 14:44-14:46) : deux postes ont servi en même temps sur la 5090 (flock -s admet N services, le .qui ne
disait pas qui avait la main). Un service porte son poste (ACVRAM_POSTE, sinon ACVRAM_SESSION) : un service d'un AUTRE
poste est refusé (4) tant qu'un service d'un poste vit ; même poste, sans poste (« - ») : partage inchangé. Hermétique :
verrou isolé par ACVRAM_VERROU, faux serveurs `sleep`, jamais les vrais /tmp/acvram-carte-*.lock."""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS", "ACVRAM_POSTE", "ACVRAM_SESSION")}
    env.update({"ACVRAM_VERROU": str(verrou), "CUDA_VISIBLE_DEVICES": "", "ACVRAM_TYPE": "service", **sup})
    return env


def _service(tmp_path, nom, duree="30", **sup):
    return subprocess.run(["bash", str(CARTE), "sleep", duree],
                          env=_env(tmp_path / "verrou.lock", ACVRAM_NOM=nom, ACVRAM_SERVICE_LOG=str(tmp_path / f"{nom}.log"), **sup),
                          capture_output=True, text=True, timeout=30)


def _tuer(*pids):
    for p in pids:
        if p and p.isdigit():
            subprocess.run(["kill", p], check=False)


def test_un_autre_poste_est_refuse_tant_que_le_premier_sert(tmp_path):
    # Casse sur l'ancien carte.sh : le second service passait (rc 0), deux postes sur la carte
    a = _service(tmp_path, "poste1-alias", ACVRAM_SESSION="poste1")
    b = None
    try:
        assert a.returncode == 0, a.stderr[-300:]
        b = _service(tmp_path, "poste6-alias", ACVRAM_SESSION="poste6")
        assert b.returncode == 4, (b.returncode, b.stdout, b.stderr[-300:])
        assert "poste1" in b.stderr and "poste1-alias" in b.stderr and "ACVRAM_POSTE=-" in b.stderr
        lignes = (tmp_path / "verrou.lock.qui").read_text().splitlines()
        assert len(lignes) == 1 and lignes[0].split()[4] == "poste=poste1", lignes
        assert "refus" in (tmp_path / "verrou.lock.journal").read_text()
    finally:
        _tuer(a.stdout.strip(), b.stdout.strip() if b else "")


def test_meme_poste_partage_comme_avant(tmp_path):
    a = _service(tmp_path, "llama-owui", ACVRAM_SESSION="poste3")
    b = _service(tmp_path, "comfyui", ACVRAM_SESSION="poste3")
    try:
        assert a.returncode == 0 and b.returncode == 0, (a.stderr[-200:], b.stderr[-200:])
        lignes = (tmp_path / "verrou.lock.qui").read_text().splitlines()
        assert [l.split()[2] for l in lignes] == ["llama-owui", "comfyui"]
        assert all(l.split()[4] == "poste=poste3" for l in lignes), lignes      # 5e champ gardé à la réécriture
    finally:
        _tuer(a.stdout.strip(), b.stdout.strip())


def test_sans_poste_ou_partage_declare_partage_avec_tous(tmp_path):
    a = _service(tmp_path, "permanent")                                     # ni ACVRAM_SESSION ni ACVRAM_POSTE
    b = _service(tmp_path, "poste1-alias", ACVRAM_SESSION="poste1")
    c = _service(tmp_path, "partage-declare", ACVRAM_SESSION="poste6", ACVRAM_POSTE="-")
    try:
        assert (a.returncode, b.returncode, c.returncode) == (0, 0, 0), (a.stderr[-200:], b.stderr[-200:], c.stderr[-200:])
        champs = [l.split() for l in (tmp_path / "verrou.lock.qui").read_text().splitlines()]
        assert [len(x) for x in champs] == [4, 5, 4], champs                    # sans poste : contrat 4 champs au bit
    finally:
        _tuer(a.stdout.strip(), b.stdout.strip(), c.stdout.strip())


def test_la_main_passe_quand_le_premier_rend_la_carte(tmp_path):
    a = _service(tmp_path, "poste1-alias", duree="2", ACVRAM_SESSION="poste1")
    assert a.returncode == 0, a.stderr[-300:]
    fin = time.time() + 15
    b = None
    while time.time() < fin:                                                # serveur mort → la ligne n'est plus vivante
        b = _service(tmp_path, "poste6-alias", ACVRAM_SESSION="poste6")
        if b.returncode == 0:
            break
        time.sleep(0.5)
    try:
        assert b is not None and b.returncode == 0, (b.returncode if b else None, b.stderr[-300:] if b else "")
    finally:
        _tuer(b.stdout.strip() if b else "")


def test_le_serveur_ne_garde_pas_le_mutex_des_postes(tmp_path):
    # un serveur qui hériterait le fd 7 bloquerait tout service suivant 30 s puis 65
    a = _service(tmp_path, "un", ACVRAM_SESSION="poste3")
    try:
        t0 = time.time()
        b = _service(tmp_path, "deux", ACVRAM_SESSION="poste3")
        assert b.returncode == 0 and time.time() - t0 < 10, (b.returncode, b.stderr[-200:])
        _tuer(b.stdout.strip())
    finally:
        _tuer(a.stdout.strip())


def test_mesure_toujours_refusee_nommee_face_a_un_service_a_poste(tmp_path):
    # Casse si une lecture du .qui laisse le 5e champ coller au type (« service poste=… ») : la mesure attendait au
    # lieu d'être refusée (carte.sh:285, trouvé par test_carte_service sous ACVRAM_SESSION)
    a = _service(tmp_path, "poste1-alias", ACVRAM_SESSION="poste1")
    try:
        t0 = time.time()
        r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                           env=_env(tmp_path / "verrou.lock", ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure", ACVRAM_ATTENTE="20"),
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 4 and time.time() - t0 < 10 and "poste1-alias" in r.stderr, (r.returncode, r.stderr[-300:])
    finally:
        _tuer(a.stdout.strip())
