"""e50.3 § 7.3 (poste4, 01/10) : `outils/campagne-qualite-e50.py --simule` — sur le parc réel
(aucune carte, REGLES §3). `--executer` n'est pas exercé ici (orchestre `qualite-e50.sh
--executer`, déjà testé de bout en bout contre un faux serveur par
`tests/test_qualite_e50_executer_faux_serveur.py`)."""
import os
import subprocess

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ICI, "outils", "campagne-qualite-e50.py")


def _parc_reel_present():
    return os.path.exists(os.path.expanduser("~/TSV/notes-modeles.tsv")) and \
        os.path.exists(os.path.expanduser("~/.kimi-code/config.toml"))


def test_simule_imprime_un_plan_et_rend_zero():
    r = subprocess.run(["python3", SCRIPT, "--simule"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "campagne-qualite-e50 --simule" in r.stdout
    assert "PRÊTE" in r.stdout
    if _parc_reel_present():
        assert "témoin acvram-qwen3-4b-srcgguf-nvfp4" in r.stdout
        assert "durée prédite totale" in r.stdout


def test_executer_refuse_sans_confirmation():
    r = subprocess.run(["python3", SCRIPT, "--executer"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 65, r.stdout + r.stderr
    assert "je-sais-que-la-carte-est-libre" in r.stderr
