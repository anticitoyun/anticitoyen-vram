"""outils/gpu/mesure/gabarit-chaine.sh (poste7-tests-30min-20-09 § 3.4) : coût croissant, sortie au premier
« faux » (code 1, PARTIEL), code 2 = partiel mais on continue, une étape sans scellé est refusée (64),
journal TSV par étape."""
from __future__ import annotations

import os
import pathlib
import subprocess

GABARIT = pathlib.Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "gabarit-chaine.sh"


def _chaine(tmp_path, corps):
    script = tmp_path / "chaine.sh"
    script.write_text(f'. "{GABARIT}"\nchaine_debut test "{tmp_path / "o"}"\n{corps}\nchaine_fin\n')
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_CARTE_TENUE"}
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)
    return r, (tmp_path / "o" / "chaine.tsv")


def test_arret_au_premier_faux_et_journal(tmp_path):
    r, tsv = _chaine(tmp_path, 'etape 0 a "s0" true\netape 1 b "s1" bash -c "exit 2"\netape 2 c "s2" false\netape 3 d "s3" true')
    assert r.returncode == 1 and "ARRÊT au premier scellé réfuté : étape 2" in r.stdout
    lignes = tsv.read_text().splitlines()
    assert [l.split("\t")[-1] for l in lignes[1:]] == ["tenu", "partiel", "faux"]      # l'étape 3 n'a jamais tourné


def test_tout_tenu_rend_zero(tmp_path):
    r, tsv = _chaine(tmp_path, 'etape 0 a "s0" true\netape 1 b "s1" true')
    assert r.returncode == 0 and "chaîne test : tenu" in r.stdout


def test_une_etape_sans_scelle_est_refusee(tmp_path):
    r, _ = _chaine(tmp_path, 'etape 0 a "" true')
    assert r.returncode == 64 and "scellé absent" in r.stdout


def test_refus_sous_un_verrou_tenu(tmp_path):
    script = tmp_path / "c.sh"; script.write_text(f'. "{GABARIT}"\n')
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=dict(os.environ, ACVRAM_CARTE_TENUE="1"))
    assert r.returncode == 3
