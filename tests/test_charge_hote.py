"""189 (chef, après la 186) : une prise `mesure` sous `outils/carte.sh` journalise la charge
hôte (`/proc/loadavg` à 1 Hz) pendant toute sa durée, dans `ACVRAM_CHARGE_DIR/<NOM>.<pid>.tsv` —
sans elle, un écart A/B (186 : 341 ms) reste invérifiable. `etat`/`partage`/`service` n'écrivent
rien (REGLES § 1) : seule `mesure` porte le plafond de durée et le risque de contention exclusive."""
from __future__ import annotations

import os
import pathlib
import subprocess

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou, charge_dir, **sup):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), ACVRAM_CHARGE_DIR=str(charge_dir),
               CUDA_VISIBLE_DEVICES="", **sup)
    return env


def test_prise_courte_ecrit_au_moins_une_ligne_de_charge(tmp_path):
    verrou = tmp_path / "v.lock"
    charge_dir = tmp_path / "charge"
    r = subprocess.run(["bash", str(CARTE), "sleep", "2"],
                       env=_env(verrou, charge_dir, ACVRAM_NOM="test-charge"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-300:]

    fichiers = list(charge_dir.glob("test-charge.*.tsv"))
    assert len(fichiers) == 1, (f"un fichier de charge attendu, vu {len(fichiers)}", r.stderr[-300:])
    lignes = fichiers[0].read_text().splitlines()
    assert len(lignes) >= 1, "aucune ligne de charge écrite pendant la prise"
    horodatage, loadavg = lignes[0].split("\t")
    assert "-" in horodatage and ":" in horodatage        # ISO, date +%FT%T
    assert len(loadavg.split()) == 3                       # 1/5/15 min, /proc/loadavg tronqué à 3


def test_type_etat_et_partage_n_ecrivent_rien(tmp_path):
    verrou = tmp_path / "v.lock"
    charge_dir = tmp_path / "charge"
    r = subprocess.run(["bash", str(CARTE), "sleep", "1"],
                       env=_env(verrou, charge_dir, ACVRAM_NOM="test-etat", ACVRAM_TYPE="etat"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-300:]
    assert not list(charge_dir.glob("test-etat.*.tsv")), (
        "ACVRAM_TYPE=etat ne doit pas journaliser la charge (REGLES § 1 : mesure seulement)")


def test_arret_garanti_sur_timeout(tmp_path):
    """La boucle de charge est tuée par le trap EXIT même sur TIMEOUT (ACVRAM_DUREE_MAX) —
    pas de process orphelin qui continue d'écrire après la restitution du verrou."""
    verrou = tmp_path / "v.lock"
    charge_dir = tmp_path / "charge"
    r = subprocess.run(["bash", str(CARTE), "sleep", "10"],
                       env=_env(verrou, charge_dir, ACVRAM_NOM="test-timeout",
                                ACVRAM_DUREE_MAX="1"),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 124, (r.returncode, r.stderr[-300:])
    fichiers = list(charge_dir.glob("test-timeout.*.tsv"))
    assert len(fichiers) == 1
    n_avant = len(fichiers[0].read_text().splitlines())
    import time
    time.sleep(2)
    n_apres = len(fichiers[0].read_text().splitlines())
    assert n_apres == n_avant, "la boucle de charge continue d'écrire après le TIMEOUT (trap EXIT rompu)"
