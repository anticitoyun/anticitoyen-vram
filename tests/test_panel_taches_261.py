"""Pièce 261 (poste2, ordre chef, reprise de la recherche de poste4, 26/09) : test à sec de
`outils/panel-taches.sh` contre un faux serveur `/v1/completions` (aucun modèle réel, aucune
carte) — vérifie la PLOMBERIE du script (deux invocations `lm_eval` séparées pour MMLU et
GSM8K, fusion des résultats, score par tâche/moyenne/pire tâche), jamais une qualité de modèle.

`lm-eval` vit dans `.venv-panel` à la racine du dépôt, séparé du `.venv` du serveur — ce test
saute (`pytest.skip`) si ce venv n'existe pas, plutôt que d'échouer faussement sur un poste qui
ne l'a pas installé (REGLES : un test qui échoue pour une raison hors de son périmètre n'est
pas un défaut du code testé).

Bras cassant : `panel-taches-resume.py` doit lever une erreur sur des résultats vides — sinon
un fichier de résultats vide donnerait silencieusement un panel « réussi » à 0 tâche.
"""
import json
import subprocess
import socket
import sys
import time
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
VENV_PANEL = RACINE / ".venv-panel" / "bin" / "python"
FAUX_SERVEUR = RACINE / "scratchpad" / "poste2-p261-26-09" / "faux-serveur.py"
RESUME = RACINE / "outils" / "panel-taches-resume.py"
PANEL = RACINE / "outils" / "panel-taches.sh"

pytestmark = pytest.mark.skipif(not VENV_PANEL.exists(), reason="lm-eval (.venv-panel) non installé sur ce poste")


def _port_libre():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_panel_taches_a_sec_contre_faux_serveur(tmp_path):
    port = _port_libre()
    serveur = subprocess.Popen([str(VENV_PANEL), str(FAUX_SERVEUR), str(port)])
    try:
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("faux serveur muet après 10 s")

        sortie = tmp_path / "sortie.json"
        r = subprocess.run(
            ["bash", str(PANEL), "faux", f"http://127.0.0.1:{port}",
             "/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4", str(sortie)],
            capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"panel-taches.sh a échoué :\n{r.stdout}\n{r.stderr}"
        d = json.loads(sortie.read_text(encoding="utf-8"))
        assert d["n_taches"] == 4
        assert set(d["taches"]) == {
            "mmlu_high_school_mathematics", "mmlu_professional_law",
            "mmlu_college_computer_science", "gsm8k"}
        assert d["taches"]["gsm8k"]["metrique"] == "exact_match,strict-match"
        assert 0.0 <= d["moyenne"] <= 1.0
        assert d["pire_tache"] in d["taches"]
        assert d["pire_score"] == min(v["score"] for v in d["taches"].values())
    finally:
        serveur.terminate()
        serveur.wait(timeout=5)


def test_le_resume_rend_faux_sur_des_resultats_vides(tmp_path):
    vide = tmp_path / "vide.json"
    vide.write_text(json.dumps({"results": {}, "configs": {}}), encoding="utf-8")
    sortie = tmp_path / "sortie.json"
    r = subprocess.run([sys.executable, str(RESUME), str(vide), str(vide), str(sortie)],
                        capture_output=True, text=True)
    assert r.returncode != 0, "un résultat vide n'aurait jamais dû rendre un panel « réussi »"
