"""Pièce 253 (chef, 26/09, `outils/carte.sh` ligne 321) : la lecture d'un
`.qui` d'une prise précédente (5e champ = pgid, pièce 244) écrivait
`read ... < "$INFO" 2>/dev/null` — l'ORDRE des redirections compte en bash :
un `<` qui échoue (fichier absent) est signalé sur stderr AVANT que le
`2>/dev/null` qui le suit dans la même ligne ne prenne effet, donc rien
n'est réellement étouffé. Le cas absent est pourtant le cas NORMAL : une
prise précédente qui s'est terminée proprement a déjà supprimé son `.qui`
(trap EXIT), donc CHAQUE prise qui suit une prise propre affichait
« <verrou>.lock.qui : Aucun fichier ou dossier de ce nom » sur stderr —
vu par chef pendant une prise de publication normale, sans rien
d'anormal en cours. Correctif : `2>/dev/null` avant `< "$INFO"`."""
from __future__ import annotations

import os
import pathlib
import subprocess

CARTE = pathlib.Path(__file__).resolve().parent.parent / "outils" / "carte.sh"


def _env(verrou):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="")
    return env


def test_premiere_prise_sans_qui_ne_bruite_pas_sur_stderr(tmp_path):
    """Aucun `.qui` n'existe encore (verrou jamais pris) : cas normal, pas
    une anomalie — stderr ne doit porter aucun message d'erreur fichier."""
    verrou = tmp_path / "acvram-carte-0.lock"
    r = subprocess.run(
        ["bash", str(CARTE), "true"],
        env=_env(verrou), capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    bruit = [l for l in r.stderr.splitlines()
             if "Aucun fichier" in l or "No such file" in l]
    assert not bruit, f"lecture du .qui absent a quand meme bruite : {bruit}"


def test_prise_qui_suit_une_prise_propre_ne_bruite_pas_non_plus(tmp_path):
    """Une prise complète et propre efface son `.qui` (trap EXIT) ; la
    SUIVANTE lit alors un fichier absent — c'est le cas réel de chef, pas
    un verrou jamais pris."""
    verrou = tmp_path / "acvram-carte-0.lock"
    r1 = subprocess.run(["bash", str(CARTE), "true"], env=_env(verrou),
                         capture_output=True, text=True, timeout=30)
    assert r1.returncode == 0, r1.stderr
    assert not (verrou.with_suffix(verrou.suffix + ".qui")).exists()

    r2 = subprocess.run(["bash", str(CARTE), "true"], env=_env(verrou),
                         capture_output=True, text=True, timeout=30)
    assert r2.returncode == 0, r2.stderr
    bruit = [l for l in r2.stderr.splitlines()
             if "Aucun fichier" in l or "No such file" in l]
    assert not bruit, f"2e prise (apres une 1ere propre) a bruite : {bruit}"
