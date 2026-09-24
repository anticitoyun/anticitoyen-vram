"""Garde d'import : un cwd dans un autre arbre acvram refuse l'import (acvram/__init__.py)."""
import os
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent


def _importer(cwd, **env_sup):
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_ARBRE_LIBRE"}
    env.update(PYTHONPATH=str(RACINE), CUDA_VISIBLE_DEVICES="", **env_sup)
    return subprocess.run([sys.executable, "-c", "import acvram"], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


def _faux_arbre(tmp_path):
    (tmp_path / "acvram").mkdir()
    (tmp_path / "acvram" / "__init__.py").write_text("")
    (tmp_path / ".git").write_text("gitdir: ailleurs\n")
    (tmp_path / "scratchpad").mkdir()
    return tmp_path


def test_cwd_dans_un_autre_arbre_refuse(tmp_path):
    # -c met "" (le cwd) en tête de sys.path : on vise le cas `python script.py`, donc cwd
    # dans un sous-dossier SANS acvram/ pour que l'import résolve vers RACINE comme en vrai.
    r = _importer(_faux_arbre(tmp_path) / "scratchpad")
    assert r.returncode != 0 and "ACVRAM_ARBRE_LIBRE" in r.stderr, r.stderr[-500:]


def test_contournement_nomme(tmp_path):
    r = _importer(_faux_arbre(tmp_path) / "scratchpad", ACVRAM_ARBRE_LIBRE="1")
    assert r.returncode == 0, r.stderr[-500:]


def test_cwd_dans_le_bon_arbre_ou_hors_arbre(tmp_path):
    assert _importer(RACINE).returncode == 0
    assert _importer(tmp_path).returncode == 0
