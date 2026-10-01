"""Pièce campagne-275 (poste2, ordre chef, 01/10) : un rc=0 de qualite.sh ne doit
plus suffire à déclarer TENU — trouvé le 01/10 (incident campagne8.log) : un `git
merge` sur le worktree de travail a cassé HEAD en plein vol, une tâche sur quatre
n'a jamais produit de résultat, et le verdict final s'est quand même imprimé
« TENU sur les 3 modèles ». `_completude` doit nommer la tâche manquante.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "outils"))
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "campagne_qualite_275",
    Path(__file__).resolve().parent.parent / "outils" / "campagne-qualite-275.py",
)
campagne275 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campagne275)

COMPLET = """\
=== PPL (X, bras=defaut)
PPL bras=6.5048 reference=6.5048 ecart_relatif=0.0 seuil=0.01 tenu=1
=== mmlu_v275_professional_law (X, bras=defaut, n=150)
mmlu_v275_professional_law : mcnemar_p=1.0 tenu=1
=== gsm8k (X, bras=defaut, n=250)
gsm8k : mcnemar_p=1.0 tenu=1
=== mmlu_v275_high_school_mathematics (X, bras=defaut, n=150)
mmlu_v275_high_school_mathematics : mcnemar_p=1.0 tenu=1
=== mmlu_v275_college_computer_science (X, bras=defaut, n=150)
mmlu_v275_college_computer_science : mcnemar_p=1.0 tenu=1
=== VERDICT (X, bras=defaut)
TENU
"""


def test_completude_sortie_complete_rien_a_signaler():
    manquantes, ppl_ok = campagne275._completude(COMPLET)
    assert manquantes == []
    assert ppl_ok is True


def test_completude_une_tache_manquante_nommee():
    # Reproduit l'incident du 01/10 : college_computer_science jamais mesurée
    # (REFUS HEAD mi-tâche), mais aucune ligne d'échec explicite derrière.
    sortie = COMPLET.replace(
        "=== mmlu_v275_college_computer_science (X, bras=defaut, n=150)\n"
        "mmlu_v275_college_computer_science : mcnemar_p=1.0 tenu=1\n",
        "",
    )
    manquantes, ppl_ok = campagne275._completude(sortie)
    assert manquantes == ["mmlu_v275_college_computer_science"]
    assert ppl_ok is True


def test_completude_toutes_taches_manquantes_ppl_seule():
    # Reproduit le cas Qwen3-4B de l'incident : seule la PPL a mesuré quelque chose.
    sortie = "=== PPL (X, bras=defaut)\nPPL bras=10.9174 reference=10.9174 ecart_relatif=0.0 seuil=0.01 tenu=1\n"
    manquantes, ppl_ok = campagne275._completude(sortie)
    assert set(manquantes) == set(campagne275.TACHES_ATTENDUES)
    assert ppl_ok is True


def test_completude_ppl_non_tenue_meme_si_taches_completes():
    sortie = COMPLET.replace("tenu=1\n", "tenu=1\n", 1).replace(
        "PPL bras=6.5048 reference=6.5048 ecart_relatif=0.0 seuil=0.01 tenu=1",
        "PPL bras=6.6 reference=6.5048 ecart_relatif=0.0177 seuil=0.01 tenu=0",
    )
    manquantes, ppl_ok = campagne275._completude(sortie)
    assert manquantes == []
    assert ppl_ok is False
