"""0.6.31 (acvram/hote.py) : réglages hôte génériques posés par le paquet — pas par la CLI seule (REGLES § 4).
Défaut = THP + OMP 8, aucune affinité ; ACVRAM_CPUS=0-3 → affinité posée et `cpus0-3` RELU sur la ligne ;
un réglage déjà posé par la session gagne (setdefault) ; la ligne porte les valeurs effectives."""
from __future__ import annotations

import os
import subprocess
import sys

from acvram import hote, regime


def _sous_processus(env_extra, code):
    env = {k: v for k, v in os.environ.items()
           if k not in ("THP_MEM_ALLOC_ENABLE", "OMP_NUM_THREADS", "ACVRAM_CPUS")}
    env.update(env_extra); env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-600:]
    return r.stdout.strip().splitlines()[-1]


def test_le_defaut_est_thp_omp8_sans_affinite():
    out = _sous_processus({}, "import acvram, os, torch; from acvram import hote; "
                              "print(hote.hote_texte(), os.environ['THP_MEM_ALLOC_ENABLE'], os.environ['OMP_NUM_THREADS'], torch.get_num_threads())")
    txt, thp, omp, nt = out.split()
    assert txt == "hote=thp,omp8" and thp == "1" and omp == "8" and nt == "8", out


def test_acvram_cpus_pose_l_affinite_et_la_ligne_la_relit():
    out = _sous_processus({"ACVRAM_CPUS": "0-3"}, "import acvram, os; from acvram import hote; "
                                                 "print(hote.hote_texte(), sorted(os.sched_getaffinity(0)))")
    assert out.startswith("hote=thp,omp8,cpus0-3 [0, 1, 2, 3]"), out


def test_un_reglage_de_session_gagne_et_la_ligne_dit_l_effectif():
    out = _sous_processus({"OMP_NUM_THREADS": "4", "THP_MEM_ALLOC_ENABLE": "0"},
                          "import acvram, torch; from acvram import hote; print(hote.hote_texte(), torch.get_num_threads())")
    assert out == "hote=sans-thp,omp4 4", out


def test_la_ligne_de_regime_a_sec_porte_hote():
    assert "hote=" in regime.regime_ligne()
    assert regime.VARIABLES and any(v.nom == "CPUS" for v in regime.VARIABLES)


def test_plages_et_cpus():
    assert hote._plages({0, 1, 2, 3, 8, 10, 11}) == "0-3,8,10-11" and hote._cpus("0-3,8") == {0, 1, 2, 3, 8}
