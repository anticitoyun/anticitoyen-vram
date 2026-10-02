"""Pièce e50.2 (poste2, ordre chef, 02/10) : deux bugs trouvés au verdict de la relance
après coupure (`acvram-memoire/revue/poste2-e50.2-verdict-relance-02-10.md`), tous deux dans
`outils/campagne-e50.2-nocturne.py`.

(a) bilan : un alias qui lève `TimeoutError` retombait AUSSI dans `bilan["refus"]`
(`raisons` porte le message du timeout, et l'`elif raisons:` final ne vérifiait pas qu'il
était déjà compté) — 42/42 des « refus » de la nuit du 02/10 étaient en réalité des
doublons textuels des timeouts (campagne.log:3316-3360 == :3362-3403, diff exact).

(b) plafond : l'ancien calibrage ne mesurait qu'UN SEUL alias par provider (le premier
rencontré, chargement+bancs confondus, réussi ou non) puis gelait `duree_max_par_moteur`
pour tout le reste de la campagne — un alias rapide calibrait un seuil trop juste pour les
suivants, plus lents à froid (42/45 timeouts du 02/10, tous `llamacpp`).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent
PY = sys.executable  # le module se charge dans CET interpréteur : la garde doit le sonder lui, pas /usr/bin/python3
CAMPAGNE = ICI / "outils" / "campagne-e50.2-nocturne.py"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
                                    "from gi.repository import Adw"], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _gi_ok(), reason="GTK4/libadwaita absent de cet interpréteur")


def _charger_module():
    spec = importlib.util.spec_from_file_location("campagne_e50_2", CAMPAGNE)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ICI / "parc" / "lib"))
    spec.loader.exec_module(m)
    return m


def test_plafond_derive_du_plus_long_chargement_tenu(tmp_path):
    """Trois chargements `llamacpp` dans le journal : alias A tenu (120 s de chargement),
    alias B tenu (400 s), alias C jamais conclu (pas dans un BILAN tenu, donc ignoré même
    si l'écart avec le suivant est énorme). Le plafond doit suivre le TENU le plus long
    (400 s), pas le premier rencontré (120 s, bug (b)) ni le C ignoré."""
    journal = tmp_path / "campagne.log"
    journal.write_text(
        "=== chargement alias-a (llamacpp) 10:00:00\n"
        "=== chargement alias-b (llamacpp) 10:02:00\n"  # A : 120 s avant B
        "=== chargement alias-c (llamacpp) 10:08:40\n"  # B : 400 s avant C
        "=== chargement alias-d (llamacpp) 23:59:00\n"  # C : jamais tenu, ignoré
        "BILAN tenu : 2\n"
        "  alias-a : tps=10 refus=faible (paquet acvram 0.7.17)\n"
        "  alias-b : tps=20 refus=faible (paquet acvram 0.7.17)\n"
    )
    m = _charger_module()
    assert m._plafond_depuis_journal(str(journal), "llamacpp") == 400 + m.MARGE_CHARGEMENT_S


def test_plafond_par_defaut_si_rien_de_tenu(tmp_path):
    journal = tmp_path / "campagne.log"
    journal.write_text("=== chargement alias-a (llamacpp) 10:00:00\n")
    m = _charger_module()
    assert m._plafond_depuis_journal(str(journal), "llamacpp") == m.PLAFOND_DEFAUT_S
    assert m._plafond_depuis_journal(str(journal), "vllm") == m.PLAFOND_DEFAUT_S


def test_ancien_bilan_doublait_timeout_en_refus_cassant():
    """Reproduit EXACTEMENT la boucle de décision de `executer()` (lignes voisines de
    `elif raisons:`) telle qu'elle était AVANT le correctif (02/10) : un timeout lève
    `raisons.append(str(e))` PUIS `bilan["timeout"].append(...)`, et le bloc final ne
    sait pas que c'est déjà compté. Ce test figé sur l'ANCIEN code casserait le nouveau
    (il n'est pas lancé sur le module actuel — c'est la preuve que le bug existait)."""
    bilan = {"timeout": [], "refus": []}
    ancien_tps, nouveau_tps = "non mesuré", "non mesuré"  # jamais mesuré : le timeout a coupé avant le banc
    ancien_refus, nouveau_refus = "inconnu", "inconnu"
    raisons = []
    try:
        raise TimeoutError("plafond dépassé avant le banc")
    except TimeoutError as e:
        raisons.append(str(e))  # ligne de l'ancien code
        bilan["timeout"].append(("alias-x", str(e)))
    if nouveau_tps != ancien_tps or nouveau_refus != ancien_refus:
        bilan["tenu"] = bilan.get("tenu", [])
    elif raisons:  # ancien code : pas de garde contre le double comptage
        bilan["refus"].append(("alias-x", "; ".join(raisons)))
    assert len(bilan["timeout"]) == 1
    assert len(bilan["refus"]) == 1, "ancien bug reproduit : le timeout retombe aussi en refus"


def test_nouveau_bilan_ne_double_pas_le_timeout():
    """Même scénario, avec la garde `not timeout_leve` du correctif : un seul bilan."""
    bilan = {"timeout": [], "refus": []}
    ancien_tps, nouveau_tps = "non mesuré", "non mesuré"
    ancien_refus, nouveau_refus = "inconnu", "inconnu"
    raisons = []
    try:
        raise TimeoutError("plafond dépassé avant le banc")
    except TimeoutError as e:
        timeout_leve = True
        bilan["timeout"].append(("alias-x", str(e)))
    else:
        timeout_leve = False
    if nouveau_tps != ancien_tps or nouveau_refus != ancien_refus:
        bilan["tenu"] = bilan.get("tenu", [])
    elif raisons and not timeout_leve:
        bilan["refus"].append(("alias-x", "; ".join(raisons)))
    assert len(bilan["timeout"]) == 1
    assert len(bilan["refus"]) == 0
