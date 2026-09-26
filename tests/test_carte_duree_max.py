"""carte.sh : plafond de prise ACVRAM_DUREE_MAX (poste7-tests-30min-20-09 § 3.1, utilisateur
07 h 17 : « aucune prise > 30 min ») — à l'échéance la commande est tuée, code 124, ligne
TIMEOUT au journal, verrou rendu ; témoin sous le plafond → 0 ; ACVRAM_TYPE=service exempté."""
from __future__ import annotations

import os
import pathlib
import subprocess
import time

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _lance(tmp_path, duree_max, cmd, type_="etat"):
    verrou = tmp_path / "verrou.lock"
    # hermétique (poste7 T4) : verrou propre ET aucun marqueur d'enveloppe hérité — sous `carte.sh pytest …` le
    # carte.sh du test refusait « DEJA tenue par cette chaine » (ACVRAM_CARTE_TENUE du parent, carte.sh:122)
    env = {k: v for k, v in os.environ.items() if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE")}
    env.update(ACVRAM_DUREE_MAX=str(duree_max), ACVRAM_VERROU=str(verrou), ACVRAM_TYPE=type_,
               ACVRAM_NOM="test-duree-max", CUDA_VISIBLE_DEVICES="")
    t0 = time.time()
    r = subprocess.run(["bash", str(CARTE), *cmd], env=env, capture_output=True, text=True, timeout=120)
    return r, time.time() - t0, verrou


def test_l_echeance_tue_la_commande_rend_124_et_le_verrou(tmp_path):
    r, dt, verrou = _lance(tmp_path, 2, ["sleep", "30"])
    assert r.returncode == 124, (r.returncode, r.stderr[-300:])
    assert dt < 20, dt                                             # SIGTERM suffit à sleep : pas d'attente des 10 s
    journal = (verrou.parent / (verrou.name + ".journal")).read_text()
    assert "TIMEOUT" in journal and "plafond=2s" in journal, journal
    assert "TIMEOUT" in r.stderr
    assert subprocess.run(["flock", "-n", str(verrou), "true"]).returncode == 0, "verrou encore tenu"


def test_le_temoin_sous_le_plafond_rend_zero(tmp_path):
    r, _, verrou = _lance(tmp_path, 5, ["sleep", "1"])
    assert r.returncode == 0, r.stderr[-300:]
    assert "TIMEOUT" not in (verrou.parent / (verrou.name + ".journal")).read_text()


def test_un_service_est_exempte(tmp_path):
    r, _, _ = _lance(tmp_path, 1, ["sleep", "2"], type_="service")
    assert r.returncode == 0, r.stderr[-300:]


def test_un_plafond_non_entier_est_refuse(tmp_path):
    r, _, _ = _lance(tmp_path, "trente", ["true"])
    assert r.returncode == 64 and "ACVRAM_DUREE_MAX" in r.stderr
