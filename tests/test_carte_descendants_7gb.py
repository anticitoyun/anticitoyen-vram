"""anticitoyen-vram-7gb (27/09 ; 30/09) : PID 248433, né juste après la prise chef-suite-076b, tenait 2,7 Gio hors
verrou au début de la mesure suivante. Le reaper c7w ne voyait que les descendants DÉJÀ dans compute-apps à la sortie ;
un descendant pas encore sur la carte lui échappait. À la sortie d'une prise mesure/état/partage, carte.sh relève
désormais tout processus vivant marqué ACVRAM_CARTE_TENUE=<la prise>, né après elle et plus sous elle, le NOMME
(journal « ORPHELIN … descendant pid … (7gb) » + stderr) et l'arrête. Un service (détaché par contrat) n'est pas visé.
Hermétique : verrou isolé (ACVRAM_VERROU dans tmp_path), processus `sleep`, aucune carte ; `recolte_carte` vérifie
qu'aucun reste ne survit au test."""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import time

import pytest

pytestmark = pytest.mark.usefixtures("recolte_carte")

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(tmp_path, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(tmp_path / "v.lock"), CUDA_VISIBLE_DEVICES="", ACVRAM_CHARGE_OK="1", **sup)
    return env


def _vivant(pid: int) -> bool:
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


@pytest.mark.parametrize("classe", ["mesure", "partage"])
def test_descendant_hors_carte_nomme_et_arrete_a_la_sortie(tmp_path, classe):
    """Cassant : sans _reaper_descendants, le `setsid sleep 30` (aucun contexte GPU, donc invisible au reaper c7w)
    survit à la prise, reparenté à systemd."""
    pidf = tmp_path / "pid"
    r = subprocess.run(["bash", str(CARTE), "bash", "-c", f"setsid sleep 30 & echo $! > {pidf}; sleep 0.3"],
                       env=_env(tmp_path, ACVRAM_TYPE=classe, ACVRAM_NOM="prise-7gb"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-400:]
    pid = int(pidf.read_text())
    assert not _vivant(pid), f"le descendant {pid} a survécu à la prise ({classe})"
    j = (tmp_path / "v.lock.journal").read_text()
    assert f"descendant pid {pid} (sleep 30 ) TERM (7gb)" in j, j[-600:]
    assert f"le descendant {pid} (sleep 30 ) survivait a la prise « prise-7gb »" in r.stderr, r.stderr[-400:]


def test_processus_etranger_non_marque_intact(tmp_path):
    etranger = subprocess.Popen(["sleep", "30"], start_new_session=True,
                                env={k: v for k, v in os.environ.items() if k != "ACVRAM_CARTE_TENUE"})
    try:
        r = subprocess.run(["bash", str(CARTE), "sleep", "0.2"], env=_env(tmp_path, ACVRAM_TYPE="mesure", ACVRAM_NOM="p"),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-400:]
        assert _vivant(etranger.pid) and "7gb" not in (tmp_path / "v.lock.journal").read_text()
    finally:
        etranger.kill(); etranger.wait()


def test_service_detache_non_vise(tmp_path):
    """Un service rend la main tout de suite par contrat (serveur détaché, verrou hérité) : jamais récolté."""
    r = subprocess.run(["bash", str(CARTE), "sleep", "30"],
                       env=_env(tmp_path, ACVRAM_TYPE="service", ACVRAM_NOM="srv", ACVRAM_SERVICE_LOG=str(tmp_path / "s.log")),
                       capture_output=True, text=True, timeout=60)
    srv = int(r.stdout.strip())
    try:
        time.sleep(0.5)
        assert _vivant(srv)
    finally:
        os.killpg(srv, signal.SIGKILL)
