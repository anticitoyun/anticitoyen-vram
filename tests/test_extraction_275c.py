"""Pièce 275c (poste2, 26/09) : bras cassant sur le nouveau motif d'extraction MMLU
(`scratchpad/poste2-p275-26-09/renoter-275c.py`) — l'ANCIEN filtre `get-answer` du banc rend les
scores officiels bas de la référence 275 (mixte-i8c) ; le NOUVEAU motif doit rendre un score
nettement plus haut sur les mêmes sorties déjà écrites (aucune carte, aucun nouveau modèle).

Saute si la référence 275 (hors git, `~/.cache/acvram/qualite-275/`) n'est pas présente sur ce
poste — ce n'est pas un défaut du code testé, juste une donnée absente ici.
"""
import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
REF = Path.home() / ".cache" / "acvram" / "qualite-275" / "Qwen3.8-27B-unsloth-mixte-i8c"
SCRIPT = RACINE / "scratchpad" / "poste2-p275-26-09" / "renoter-275c.py"

pytestmark = pytest.mark.skipif(not REF.exists(), reason="référence 275 (hors git) absente de ce poste")

_TACHES = [
    "mmlu_flan_cot_fewshot_high_school_mathematics",
    "mmlu_flan_cot_fewshot_professional_law",
    "mmlu_flan_cot_fewshot_college_computer_science",
]
# scores officiels HISTORIQUES (mixte-i8c, ancien filtre `get-answer`, 275b) — fixés en dur :
# `panel.json` de la référence a été mis à jour en 275d avec les scores CORRIGÉS sous les
# nouveaux noms de tâche (`mmlu_v275_*`) ; ce test compare l'ancien comportement à sa valeur
# HISTORIQUE, plus au fichier de référence (qui ne la porte plus, par construction).
_SCORE_OFFICIEL_ANCIEN = {
    "mmlu_flan_cot_fewshot_high_school_mathematics": 0.2,
    "mmlu_flan_cot_fewshot_professional_law": 0.08,
    "mmlu_flan_cot_fewshot_college_computer_science": 0.11,
}


def _samples(tache):
    d = REF / f"{tache}.json.echantillons" / "coder"
    fichiers = sorted(d.glob(f"samples_{tache}_*.jsonl"))
    assert fichiers, f"aucun jsonl pour {tache} sous {d}"
    return fichiers[0]


def test_ancien_filtre_reproduit_le_score_officiel():
    for tache in _TACHES:
        import subprocess
        out = subprocess.run([sys.executable, str(SCRIPT), str(_samples(tache)), "ancien"],
                              capture_output=True, text=True, check=True).stdout
        score = json.loads(out.split("RESULTAT_RENOTATION ", 1)[1])["score"]
        assert abs(score - _SCORE_OFFICIEL_ANCIEN[tache]) < 1e-6, (
            f"{tache} : ancien filtre {score} != score officiel historique {_SCORE_OFFICIEL_ANCIEN[tache]}")


def test_nouveau_filtre_rend_un_score_nettement_meilleur():
    for tache in _TACHES:
        import subprocess
        out = subprocess.run([sys.executable, str(SCRIPT), str(_samples(tache)), "nouveau"],
                              capture_output=True, text=True, check=True).stdout
        score = json.loads(out.split("RESULTAT_RENOTATION ", 1)[1])["score"]
        assert score >= 0.5, f"{tache} : nouveau motif {score} < 0,5 (prédiction 275c non tenue)"
