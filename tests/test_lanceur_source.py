"""acvram-serveur (parc/bin) à sec : paquet par défaut, opt-in ACVRAM_ARBRE, ligne « source= » à deux valeurs.
Rien n'est servi : ACVRAM_SERVEUR_A_SEC=1 arrête le lanceur juste avant `serve` ; faux binaires dans un tmp ; port 8090 non touché
(le lanceur sonde /v1/models par curl : ici PORT est celui d'un port fermé pour ne rien tuer)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

LANCEUR = Path(__file__).resolve().parent.parent / "parc" / "bin" / "acvram-serveur"


@pytest.fixture
def poste(tmp_path):
    home = tmp_path / "home"; (home / "TSV").mkdir(parents=True)
    modele = tmp_path / "Modele-nvfp4"; modele.mkdir(); (modele / "config.json").write_text("{}")
    (home / "TSV" / "acvram-chemins.tsv").write_text(f"acvram-essai\t{modele}\t32768\n")
    paquet = tmp_path / "paquet-acvram"; paquet.write_text("#!/bin/sh\n[ \"$1\" = --version ] && echo 'acvram 9.9.9'\n"); paquet.chmod(0o755)
    arbre = tmp_path / "arbre"; (arbre / ".venv" / "bin").mkdir(parents=True); (arbre / "acvram").mkdir()
    (arbre / ".venv" / "bin" / "acvram").write_text("#!/bin/sh\necho arbre\n"); (arbre / ".venv" / "bin" / "acvram").chmod(0o755)
    subprocess.run(["git", "init", "-q", "-b", "main", str(arbre)], check=True)
    subprocess.run(["git", "-C", str(arbre), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    # le lanceur lit ~/TSV et tue ce qui écoute sur $PORT : HOME factice, port fermé
    src = LANCEUR.read_text().replace("PORT=8090", "PORT=1")
    lanceur = tmp_path / "acvram-serveur"; lanceur.write_text(src); lanceur.chmod(0o755)
    env = {**os.environ, "HOME": str(home), "ACVRAM_SERVEUR_A_SEC": "1", "ACVRAM_PAQUET_BIN": str(paquet)}
    env.pop("ACVRAM_ARBRE", None)
    return {"lanceur": lanceur, "env": env, "arbre": arbre, "paquet": paquet}


def _run(p, **sup):
    return subprocess.run(["bash", str(p["lanceur"]), "acvram-essai"], capture_output=True, text=True, env={**p["env"], **sup}, timeout=60)


def test_paquet_par_defaut(poste):
    r = _run(poste)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acvram : source=paquet(9.9.9)" in r.stdout
    assert f"commande : {poste['paquet']} serve" in r.stdout and "--served-name acvram-essai" in r.stdout


def test_arbre_opt_in_propre_puis_sale(poste):
    sha = subprocess.run(["git", "-C", str(poste["arbre"]), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    r = _run(poste, ACVRAM_ARBRE=str(poste["arbre"]))
    assert r.returncode == 0 and f"acvram : source=arbre(main@{sha},propre)" in r.stdout, r.stdout + r.stderr
    assert f"commande : {poste['arbre']}/.venv/bin/acvram serve" in r.stdout
    (poste["arbre"] / "acvram" / "x.py").write_text("# modif non commitée\n")
    r2 = _run(poste, ACVRAM_ARBRE=str(poste["arbre"]))
    assert f"source=arbre(main@{sha},sale)" in r2.stdout, r2.stdout


def test_faute_construite_paquet_absent_et_arbre_sans_venv(poste, tmp_path):
    r = _run(poste, ACVRAM_PAQUET_BIN=str(tmp_path / "absent"))
    assert r.returncode == 1 and "paquet acvram absent" in r.stderr
    r2 = _run(poste, ACVRAM_ARBRE=str(tmp_path / "pas-un-depot"))
    assert r2.returncode == 1 and "pas de .venv/bin/acvram" in r2.stderr


def test_a_sec_ne_lance_rien(poste):
    """Aucun processus « serve » ne survit au lanceur à sec."""
    _run(poste)
    ps = subprocess.run(["pgrep", "-af", "acvram-essai"], capture_output=True, text=True).stdout
    assert "serve" not in ps, ps
