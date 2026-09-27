"""276 f : `ACVRAM_ARBRE=<arbre>` — l'import d'acvram depuis un autre arbre est refusé à la source, et l'arbre importé est
imprimé (« ARBRE <chemin> ») pour que toute prise A/B entre deux arbres le contrôle. Sans la variable : rien ne change."""
import os
import subprocess
import sys

import acvram

RACINE = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(acvram.__file__))))


def _import(env_sup, cwd=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_ARBRE")}
    env.update(env_sup)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-c", "import acvram; print('OK')"], cwd=cwd or RACINE, env=env,
                          capture_output=True, text=True, timeout=180)


def test_arbre_demande_egal_a_l_arbre_importe_imprime_la_ligne():
    r = _import({"ACVRAM_ARBRE": RACINE})
    assert r.returncode == 0, r.stderr[-800:]
    assert f"ARBRE {RACINE}" in r.stdout.splitlines() and "OK" in r.stdout


def test_arbre_demande_different_refuse_a_l_import(tmp_path):
    r = _import({"ACVRAM_ARBRE": str(tmp_path)})
    assert r.returncode != 0
    assert "ACVRAM_ARBRE demande" in r.stderr and str(tmp_path.resolve()) in r.stderr


def test_sans_variable_rien_ne_change():
    r = _import({})
    assert r.returncode == 0 and "ARBRE " not in r.stdout and "OK" in r.stdout


def test_arbre_libre_desarme_aussi_cette_garde(tmp_path):
    r = _import({"ACVRAM_ARBRE": str(tmp_path), "ACVRAM_ARBRE_LIBRE": "1"})
    assert r.returncode == 0, r.stderr[-800:]
