"""anticitoyen-vram-jxm (ticket resté ouvert) : le trap générique EXIT de carte.sh
(TYPE=mesure|etat) faisait `rm -f "$INFO"` sans condition, comme le gardien de service
avant son correctif. Même garde par symétrie (défense en profondeur — flock devrait déjà
empêcher un autre détenteur d'écrire pendant que celui-ci tient le verrou), et une trace
ANOMALIE au journal si jamais ce trap trouvait un `.qui` qui n'est plus le sien, pour
laisser une preuve la prochaine fois plutôt qu'un silence. Tests à sec, verrou isolé
(ACVRAM_VERROU) : jamais les vrais /tmp/acvram-carte-*.lock du circuit."""
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


def test_mesure_normale_efface_son_qui_et_journalise_rendue(tmp_path):
    """Contrôle : sans tripatouillage, le comportement d'avant est inchangé —
    le .qui disparaît à la fin, une ligne « rendue » au journal."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    journal = tmp_path / "verrou.lock.journal"
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure", ACVRAM_DUREE_MAX="0"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-300:]
    assert not info.exists(), ".qui toujours présent après une mesure normale"
    assert " rendue " in journal.read_text() and "ANOMALIE" not in journal.read_text()


def test_trap_generique_n_efface_pas_un_qui_deja_repris(tmp_path):
    """Pendant qu'une mesure tourne, son `.qui` est remplacé par celui d'un AUTRE
    détenteur (simulation d'un cas anormal — flock devrait l'empêcher en service normal,
    mais le trap doit rester sûr par lui-même, symétrique au gardien de service jxm) :
    à la sortie, le trap ne doit PAS effacer ce fichier qui n'est plus le sien, et doit
    le dire au journal (ANOMALIE), pas rester silencieux."""
    verrou = tmp_path / "verrou.lock"
    info = tmp_path / "verrou.lock.qui"
    journal = tmp_path / "verrou.lock.journal"
    proc = subprocess.Popen(["bash", str(CARTE), "sleep", "2"],
                            env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure", ACVRAM_DUREE_MAX="0"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # laisser carte.sh écrire son .qui et poser son trap
    deadline = time.time() + 5
    while time.time() < deadline and not info.exists():
        time.sleep(0.05)
    assert info.exists(), ".qui jamais apparu"
    info.write_text("999999 0 un-autre-detenteur mesure\n")
    proc.wait(timeout=30)
    assert proc.returncode == 0
    assert info.exists() and info.read_text().split()[0] == "999999", \
        ".qui d'un autre détenteur effacé par le trap générique"
    lignes = journal.read_text()
    assert "ANOMALIE" in lignes and "ma-mesure" in lignes and "999999" in lignes, lignes
