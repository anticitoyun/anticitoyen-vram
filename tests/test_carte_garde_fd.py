"""anticitoyen-vram-575 (trouvé en écrivant les tests de jxm) : le garde-fou de durée
(`ACVRAM_DUREE_MAX`, outils/carte.sh) héritait stdout/stderr de l'appelant sans jamais les
fermer, et son `kill` ne tuait que la sous-shell, pas le `sleep "$DUREE_MAX"` qu'elle avait
lancé — orphelin, gardant les descripteurs jusqu'à l'expiration du délai. Un appelant qui
CAPTURE la sortie (`subprocess.run(capture_output=True)`) attendait alors l'EOF du tuyau
jusqu'à DUREE_MAX (1800 s par défaut), même quand la commande finissait en une seconde.
Tests à sec, verrou isolé (ACVRAM_VERROU) : jamais les vrais /tmp/acvram-carte-*.lock."""
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


def test_capture_de_sortie_ne_bloque_pas_jusqu_a_duree_max(tmp_path):
    """Le cassant : DUREE_MAX bien supérieur au temps réel de la commande — sans le
    correctif, `communicate()` pend jusqu'à DUREE_MAX (30 s ici, déjà largement assez pour
    dépasser « quelques secondes ») au lieu de rendre la main dès que « sleep 1 » finit."""
    verrou = tmp_path / "verrou.lock"
    t0 = time.time()
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure",
                                ACVRAM_DUREE_MAX="30"),
                       capture_output=True, text=True, timeout=15)
    dt = time.time() - t0
    assert r.returncode == 0, r.stderr[-300:]
    assert dt < 5, f"la capture de sortie a attendu {dt:.1f}s (garde-fou DUREE_MAX=30 non fermé)"


def _sleep_30_orphelins():
    """PIDs de tout `sleep 30` déjà reparenté à init (PPID 1) — la signature exacte
    d'un garde-fou dont la sous-shell est morte sans emporter son enfant."""
    r = subprocess.run(["pgrep", "-f", "^sleep 30$"], capture_output=True, text=True).stdout.split()
    out = []
    for pid in r:
        ppid = subprocess.run(["ps", "-o", "ppid=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if ppid == "1":
            out.append(pid)
    return set(out)


def test_aucun_sleep_orphelin_apres_la_commande(tmp_path):
    """Le `sleep DUREE_MAX` du garde-fou ne doit pas survivre à la commande qu'il
    surveillait : `pkill -P "$_garde"` doit l'atteindre, pas seulement la sous-shell."""
    verrou = tmp_path / "verrou.lock"
    avant = _sleep_30_orphelins()
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, ACVRAM_TYPE="mesure", ACVRAM_NOM="ma-mesure",
                                ACVRAM_DUREE_MAX="30"),
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0, r.stderr[-300:]
    time.sleep(1)   # laisser un éventuel reparentage vers init se stabiliser
    nouveaux = _sleep_30_orphelins() - avant
    assert not nouveaux, f"sleep 30 orphelin(s) laissé(s) par le garde-fou : {nouveaux}"
