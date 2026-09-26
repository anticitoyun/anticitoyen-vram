"""anticitoyen-vram-c7w : un `acvram serve` lancé par la commande sous `setsid`
(sa propre session) survivait au TERM/KILL de la commande — la 115 bis de poste2,
rendue par TIMEOUT le 23/09 à 23h00, a laissé 28,5 Gio occupés et fait tomber la
prise suivante de poste5 en OOM (`pkill -P` rate un enfant reparenté ou lancé
depuis un sous-shell mort avant lui). Correctif : `_reaper_setsid_orphelins`
repère par `/proc/<pid>/environ` (marque `ACVRAM_CARTE_TENUE=<pid détenteur>`,
héritée quel que soit le reparentage) tout compute-app orphelin de LA CHAÎNE,
TERM puis KILL, journalise, avant de rendre — dans le chemin TIMEOUT et dans
le trap générique. `nvidia-smi` factice (PATH), aucune carte réelle requise."""
from __future__ import annotations

import os
import pathlib
import stat
import subprocess
import time

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _faux_nvidia_smi(tmp_path, pids_file):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / "nvidia-smi"
    f.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        '  *query-compute-apps=pid*) cat "' + str(pids_file) + '" 2>/dev/null ;;\n'
        "  *) echo '0,0,0' ;;\n"
        "esac\n"
    )
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return d


def _env(verrou, path_bin, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="",
               PATH=f"{path_bin}:{env['PATH']}", **sup)
    return env


def test_orphelin_setsid_est_acheve_au_timeout(tmp_path):
    verrou = tmp_path / "verrou.lock"
    journal = tmp_path / "verrou.lock.journal"
    pids_file = tmp_path / "compute-apps.csv"
    orphan_pidfile = tmp_path / "orphan.pid"
    bin_dir = _faux_nvidia_smi(tmp_path, pids_file)

    # La « commande » : lance un faux serveur détaché (setsid, propre session),
    # hérite ACVRAM_CARTE_TENUE sans rien faire de spécial, puis dort au-delà
    # du plafond — TIMEOUT tue la commande, pas son petit-fils setsid.
    cmd = tmp_path / "cmd.sh"
    # Le setsid part dans un SOUS-SHELL qui rend la main tout de suite : le
    # setsid est orphelin (reparenté à init) BIEN AVANT le TIMEOUT — comme la
    # 115 bis, `pkill -P $_fils` (enfants DIRECTS du process tué) ne le voit
    # plus, seule la marque ACVRAM_CARTE_TENUE dans son environ le retrouve.
    cmd.write_text(
        "#!/bin/bash\n"
        f'( setsid sleep 60 < /dev/null > /dev/null 2>&1 & echo $! | tee "{orphan_pidfile}" > "{pids_file}" )\n'
        "sleep 30\n"
    )
    cmd.chmod(cmd.stat().st_mode | stat.S_IEXEC)

    r = subprocess.run(["bash", str(CARTE), "bash", str(cmd)],
                       env=_env(verrou, bin_dir, ACVRAM_TYPE="mesure", ACVRAM_NOM="c7w-test", ACVRAM_DUREE_MAX="2"),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 124, (r.returncode, r.stderr[-500:])

    orphan_pid = int(orphan_pidfile.read_text().strip())

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(orphan_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"orphelin setsid {orphan_pid} encore vivant après TIMEOUT — c7w non corrigé")

    lignes = journal.read_text()
    assert "ORPHELIN" in lignes and str(orphan_pid) in lignes, lignes
