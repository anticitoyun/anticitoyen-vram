"""bd jdp (chef, 01/10) : la garde d'import `acvram/__init__.py:_garde_arbre` (276 f) refuse
un import depuis un AUTRE arbre que celui que `ACVRAM_ARBRE` demande — mais si rien ne pose
`ACVRAM_ARBRE` ET que le cwd n'est dans AUCUN arbre acvram (scratchpad, `/tmp`), elle laisse
passer en silence. Le trou réel (poste1, 01/10, `outils/gpu/mesure/ppl-balayage-11e.sh`) : un
venv principal éditable, lancé depuis un worktree dont le cwd n'est pas l'arbre, importe
l'acvram de l'arbre PRINCIPAL sans qu'aucune garde ne le signale.

Correctif (v1) : `outils/carte.sh` (aux six points où il lance la commande) et chaque script de
`outils/gpu/mesure/*.sh` qui lance python exportent `ACVRAM_ARBRE="${ACVRAM_ARBRE:-<leur arbre>}"`
— jamais d'écrasement si la chaîne appelante (ABBA à deux arbres) l'a déjà posée.

Correctif (v2, chef après relecture) : `<leur arbre>` fixé au dépôt qui PORTE le script cassait
un usage réel — des scripts appellent le carte.sh d'un AUTRE arbre depuis leur propre worktree
(`scratchpad/poste1-p221-25-09/chaine.sh:5`, `~/.config/acvram/chef/fenetre-g2c.sh`) : avec
`ACVRAM_ARBRE=$DEPOT` fixe, le cwd de l'appelant (son propre worktree) ne correspond plus à
l'arbre importé, et l'import est refusé À TORT (refus bruyant, jamais un faux positif silencieux,
mais l'usage casse). `outils/arbre-defaut.sh:_acvram_arbre_defaut` reproduit la remontée de
`_garde_arbre` : l'arbre qui CONTIENT le cwd prime, `$DEPOT` (l'arbre qui porte le script)
seulement si le cwd n'est dans aucun arbre.

Deux tests cassants : (1) cwd hors de tout arbre + interpréteur d'un AUTRE arbre réel → refus
(piège d'origine, toujours valide : repli sur $DEPOT) ; (2) `carte.sh` de CE dépôt appelé avec un
cwd dans l'arbre PRINCIPAL réel → importe le PRINCIPAL, PAS refusé (piège v2 : l'arbre du cwd
doit primer sur $DEPOT). Avant le correctif v1, (1) passait l'import EN SILENCE ; avant le
correctif v2, (2) était refusé À TORT (vérifié par retrait, voir le verdict)."""
import os
import re
import pathlib
import subprocess
import sys
import pytest

pytestmark = pytest.mark.usefixtures("recolte_carte")

RACINE = pathlib.Path(__file__).resolve().parent.parent
CARTE = RACINE / "outils" / "carte.sh"
MESURE = RACINE / "outils" / "gpu" / "mesure"

AUTRE_ARBRE = pathlib.Path(os.path.expanduser("~/Bureau/Claude/anticitoyen-vram"))
AUTRE_PY = AUTRE_ARBRE / ".venv" / "bin" / "python"


def _env(verrou, cwd_hors_arbre):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS",
                        "ACVRAM_ARBRE", "PYTHONPATH")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="")
    return env


@pytest.mark.skipif(not AUTRE_PY.is_file(), reason="venv principal absent sur ce poste")
def test_carte_sh_refuse_un_acvram_dune_autre_racine(tmp_path):
    """Le piège réel : cwd hors de tout arbre (tmp_path), interpréteur du venv PRINCIPAL (qui
    importe acvram depuis SA racine, différente de ce dépôt-ci si on tourne dans un worktree) —
    `carte.sh` doit maintenant poser ACVRAM_ARBRE=<ce dépôt> et faire échouer l'import."""
    if AUTRE_ARBRE.resolve() == RACINE.resolve():
        pytest.skip("ce test tourne déjà dans l'arbre principal — rien à opposer")
    verrou = tmp_path / "acvram-carte-0.lock"
    r = subprocess.run(
        ["bash", str(CARTE), str(AUTRE_PY), "-c", "import acvram; print('IMPORT OK')"],
        cwd=str(tmp_path), env=_env(verrou, tmp_path),
        capture_output=True, text=True, timeout=30)
    assert r.returncode != 0, f"import accepté en silence : {r.stdout!r} {r.stderr!r}"
    assert "ACVRAM_ARBRE demande" in r.stderr, r.stderr
    assert "IMPORT OK" not in r.stdout


@pytest.mark.skipif(not AUTRE_PY.is_file(), reason="venv principal absent sur ce poste")
def test_carte_sh_dun_arbre_importe_lacvram_du_cwd_dans_un_autre_arbre(tmp_path):
    """Piège v2 (chef) : `outils/carte.sh` DE CE DÉPÔT, cwd dans l'arbre PRINCIPAL réel —
    l'arbre du cwd doit primer sur $DEPOT (le dépôt qui porte ce carte.sh) : import accepté,
    acvram importé depuis l'arbre PRINCIPAL, jamais refusé."""
    if AUTRE_ARBRE.resolve() == RACINE.resolve():
        pytest.skip("ce test tourne déjà dans l'arbre principal — rien à opposer")
    verrou = tmp_path / "acvram-carte-0.lock"
    env = {k: v for k, v in os.environ.items()
          if k not in ("ACVRAM_CARTE_TENUE", "ACVRAM_VERROU", "ACVRAM_CARTE", "ACVRAM_CPUS",
                       "ACVRAM_ARBRE", "PYTHONPATH")}
    env.update(ACVRAM_VERROU=str(verrou), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        ["bash", str(CARTE), str(AUTRE_PY), "-c",
         "import acvram, os; print('IMPORT', os.path.dirname(os.path.dirname(acvram.__file__)))"],
        cwd=str(AUTRE_ARBRE), env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"refusé à tort (cwd dans l'arbre principal) : {r.stderr}"
    assert f"IMPORT {AUTRE_ARBRE}" in r.stdout, r.stdout


def test_le_controle_peut_rendre_faux(tmp_path):
    """Bras cassant direct : sans ACVRAM_ARBRE du tout, le même import passe EN SILENCE — prouve
    que le contrôle ci-dessus détecte bien l'absence du correctif, pas une coïncidence d'environnement."""
    if not AUTRE_PY.is_file() or AUTRE_ARBRE.resolve() == RACINE.resolve():
        pytest.skip("venv principal absent ou déjà l'arbre courant")
    env = {k: v for k, v in os.environ.items() if k not in ("ACVRAM_ARBRE", "PYTHONPATH")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([str(AUTRE_PY), "-c", "import acvram; print('IMPORT OK')"],
                       cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "IMPORT OK" in r.stdout, (
        "le bras témoin (sans ACVRAM_ARBRE) devrait laisser passer l'import en silence — "
        f"sinon ce test ne prouve plus que le correctif est nécessaire : {r.stderr}")


# ---- les scripts .sh de outils/gpu/mesure qui lancent python exportent ACVRAM_ARBRE ----

MOTIF_LANCE_PYTHON = re.compile(r'"\$PY[A-Z_]*"|\$\{PY[A-Z_]*\}|\bpython3?\b', re.IGNORECASE)
MOTIF_ARBRE = re.compile(r"ACVRAM_ARBRE=")


def _scripts_qui_lancent_python():
    for chemin in sorted(MESURE.glob("*.sh")):
        texte = chemin.read_text(encoding="utf-8", errors="replace")
        corps = "\n".join(l for l in texte.splitlines() if not l.lstrip().startswith("#"))
        if MOTIF_LANCE_PYTHON.search(corps):
            yield chemin.name, texte


def test_chaque_script_de_mesure_qui_lance_python_exporte_acvram_arbre():
    manquants = [nom for nom, texte in _scripts_qui_lancent_python() if not MOTIF_ARBRE.search(texte)]
    assert not manquants, (
        "ces scripts de outils/gpu/mesure/ lancent python sans jamais poser ACVRAM_ARBRE : "
        f"le piège 01/10 (venv principal éditable) leur reste ouvert — {manquants}")
