"""e50.3 § 7 (poste3, 01/10) : `outils/qualite-e50.sh` en `--simule` (défaut, REGLES §3 : aucune
carte, aucun processus) — résout un alias réel du parc, détecte « sans raisonnement » (e50.1),
refuse un alias inconnu, refuse `--executer` sans la confirmation explicite. Tourne sur le parc
réel (`~/TSV`), aucune prise (aucune carte ni lm-eval ne sont lancés par `--simule`)."""
import os
import re
import shutil
import subprocess
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ICI, "outils", "qualite-e50.sh")


def _gi_ok():
    r = subprocess.run(["/usr/bin/python3", "-c",
                       "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


def _un_alias_reel(thinking_attendu):
    """Un alias réel du parc : `acvram-qwen3.8-27b-nvfp4` est connu thinking=True (gabarit
    Qwen3, e50.1) ; `acvram-deepseek-coder-6-7b-nvfp4` connu thinking=False (pas de <think>)."""
    return "acvram-qwen3.8-27b-nvfp4" if thinking_attendu else "acvram-deepseek-coder-6-7b-nvfp4"


def _parc_reel_a_ces_alias():
    tsv = os.path.expanduser("~/TSV/acvram-chemins.tsv")
    try:
        aliases = {l.split("\t")[0] for l in open(tsv) if l.strip() and not l.startswith("#")}
    except OSError:
        return False
    return {_un_alias_reel(True), _un_alias_reel(False)} <= aliases


pytestmark = pytest.mark.skipif(not _gi_ok() or not _parc_reel_a_ces_alias(),
                                reason="GTK4/libadwaita ou alias de référence absents du parc réel")


def test_simule_resout_un_alias_thinking():
    r = subprocess.run(["bash", SCRIPT, _un_alias_reel(True)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "thinking  : True" in r.stdout, r.stdout
    assert "sans raisonnement si possible : oui" in r.stdout, r.stdout
    assert "rien lancé" in r.stdout


def test_simule_resout_un_alias_non_thinking():
    r = subprocess.run(["bash", SCRIPT, _un_alias_reel(False)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "thinking  : False" in r.stdout, r.stdout
    assert "sans raisonnement si possible : non" in r.stdout, r.stdout


def test_refuse_un_alias_inconnu():
    r = subprocess.run(["bash", SCRIPT, "alias-qui-nexiste-pas-du-tout"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUS" in r.stderr


def test_refuse_executer_sans_confirmation():
    r = subprocess.run(["bash", SCRIPT, _un_alias_reel(True), "--executer"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 65, r.stdout + r.stderr
    assert "je-sais-que-la-carte-est-libre" in r.stderr


def test_refuse_argument_inconnu():
    r = subprocess.run(["bash", SCRIPT, _un_alias_reel(True), "--mode-bidon"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 64, r.stdout + r.stderr


# ---- pin : les limites par tâche ne changent pas sans qu'on les voie (REGLES) ----

def test_limites_par_tache_pin():
    """Les limites (30/30/30/40/40) sont des nombres en dur dans le script, jamais une variable
    surprise — ce test les lit dans le texte affiché par --simule et casse si quelqu'un les
    change : un changement est alors visible ICI, pas seulement dans le script."""
    r = subprocess.run(["bash", SCRIPT, _un_alias_reel(True)], capture_output=True, text=True, timeout=30)
    ligne = next(l for l in r.stdout.splitlines() if l.startswith("tâches"))
    attendu = ("tâches    : mmlu_e50_hsm(30) mmlu_e50_law(30) mmlu_e50_ccs(30) "
              "gsm8k_e50(40) humaneval_e50(40)")
    assert ligne == attendu, f"limites changées sans que ce test le voie : {ligne!r}"
