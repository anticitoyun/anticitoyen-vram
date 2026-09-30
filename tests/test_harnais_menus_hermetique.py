"""Tout test qui LANCE outils/test-menus-reels.py (sous-processus) coupe l attente « carte vide » du harnais
(`TMR_CARTE=""`, 249674e39) : sinon, sous une prise GPU (pytest tient un contexte CUDA sur la carte), chaque bras attend
120 s pour de vrai — test_contenu_final_syy en TimeoutExpired dans la suite GPU de la 0.7.16 (chef, 30/09).
Garde de source : casse si un test lance le harnais sans poser TMR_CARTE, et prouve que l attente est bien coupée
quand la carte paraît occupée (faux nvidia-smi : un PID de calcul, 20 000 Mio)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
OUTIL = TESTS.parent / "outils" / "test-menus-reels.py"


def test_tout_lanceur_du_harnais_coupe_l_attente_carte():
    fautifs = []
    for f in sorted(TESTS.glob("test_*.py")):
        if f.name == Path(__file__).name:
            continue
        s = f.read_text(encoding="utf-8")
        lance = "test-menus-reels.py" in s and "subprocess" in s and "OUTIL" in s   # OUTIL = le harnais, lancé
        if lance and '"TMR_CARTE": ""' not in s:
            fautifs.append(f.name)
    assert not fautifs, f"lancent le harnais sans TMR_CARTE=\"\" : {fautifs}"


def test_carte_occupee_sans_attente_quand_coupee(tmp_path):
    """Le harnais, carte « occupée » (faux nvidia-smi), TMR_CARTE="" : aucune attente ; TMR_CARTE=0 : il attend."""
    b = tmp_path / "bin"; b.mkdir()
    (b / "nvidia-smi").write_text("#!/bin/sh\ncase \"$*\" in *compute-apps*) echo 424242;; *memory.used*) echo 20000;; esac\n")
    (b / "nvidia-smi").chmod(0o755)
    code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('t',sys.argv[1]);m=importlib.util.module_from_spec(s);"
            "s.loader.exec_module(m);m.est_permanent=lambda p:False;print(repr(m.attendre_carte_libre(delai=3,pas=0.5)))")
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}"}
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-c", code, str(OUTIL)], env={**env, "TMR_CARTE": ""}, capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0 and r.stdout.strip() == "''" and time.monotonic() - t0 < 10, r.stdout + r.stderr
    r = subprocess.run([sys.executable, "-c", code, str(OUTIL)], env={**env, "TMR_CARTE": "0"}, capture_output=True,
                       text=True, timeout=30)
    assert "NON libérée en 3 s" in r.stdout, r.stdout + r.stderr    # le faux nvidia-smi est bien lu : le test peut rendre faux
