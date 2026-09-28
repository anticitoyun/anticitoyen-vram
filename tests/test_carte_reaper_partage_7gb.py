"""anticitoyen-vram-7gb (27/09, ordre chef) : un descendant GPU d'une prise
PARTAGE pouvait survivre à sa restitution et contaminer la mesure suivante
(même famille que les orphelins `carte.sh sleep N` de ked — un `setsid`
reparenté avant que la commande ne meure). La classe partage n'appelait
jamais `_reaper_setsid_orphelins` à la sortie, contrairement à mesure/etat
(c7w). Corrigé : le trap EXIT de la classe partage relève désormais
`nvidia-smi --query-compute-apps` et achève tout descendant marqué
`ACVRAM_CARTE_TENUE=<pid de la prise>`, comme mesure. `nvidia-smi` factice
(PATH), aucune carte réelle requise."""
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


def test_partage_acheve_son_orphelin_gpu_a_la_sortie(tmp_path):
    verrou = tmp_path / "verrou.lock"
    journal = tmp_path / "verrou.lock.journal"
    pids_file = tmp_path / "compute-apps.csv"
    orphan_pidfile = tmp_path / "orphan.pid"
    bin_dir = _faux_nvidia_smi(tmp_path, pids_file)

    # La commande lance un setsid détaché (hérite ACVRAM_CARTE_TENUE) puis
    # rend la main tout de suite : la commande de carte.sh se termine avant
    # que le petit-fils ne meure, exactement le cas 7gb (fin de prise normale,
    # pas un TIMEOUT — la classe partage n'a pas de DUREE_MAX).
    cmd = tmp_path / "cmd.sh"
    cmd.write_text(
        "#!/bin/bash\n"
        f'setsid sleep 60 < /dev/null > /dev/null 2>&1 & echo $! | tee "{orphan_pidfile}" > "{pids_file}"\n'
        "exit 0\n"
    )
    cmd.chmod(cmd.stat().st_mode | stat.S_IEXEC)

    r = subprocess.run(["bash", str(CARTE), "bash", str(cmd)],
                       env=_env(verrou, bin_dir, ACVRAM_TYPE="partage", ACVRAM_NOM="7gb-test"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (r.returncode, r.stderr[-500:])

    orphan_pid = int(orphan_pidfile.read_text().strip())

    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(orphan_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"orphelin GPU {orphan_pid} d'un partage encore vivant — 7gb non corrigé")

    lignes = journal.read_text()
    assert "ORPHELIN" in lignes and str(orphan_pid) in lignes, lignes
