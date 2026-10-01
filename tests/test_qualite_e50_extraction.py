"""e50.3 § 7 (poste3, 01/10) : le filtre `get-answer-v275` des yaml e50 (identique au filtre
275d, voir `outils/lm_eval_taches/mmlu_e50_hsm.yaml`) rejoue CORRECTEMENT 20 conclusions
RÉELLES du dépôt (`tests/fixtures/echantillons_mmlu_e50_20.json`, extraites de
`~/.cache/acvram/qualite-275/*/mmlu_v275_*.json.echantillons/`) — comparées à ce que lm-eval
lui-même avait extrait (`filtered_resps`) lors de la campagne 275, qui utilise EXACTEMENT le
même `regex_pattern`/`mapping_dict`. Aucun lm-eval requis ici : la regex est rejouée en Python
pur, fidèle au yaml."""
import json
import os
import re

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ICI, "tests", "fixtures", "echantillons_mmlu_e50_20.json")

# Copie exacte du filter_list « get-answer-v275 » des yaml (regex, uppercase, map).
_REGEX = re.compile(r"answer[^A-D]{0,25}([A-D])", re.IGNORECASE)
_MAPPING = {"A": "(A)", "B": "(B)", "C": "(C)", "D": "(D)"}


def get_answer_v275(texte):
    correspondances = _REGEX.findall(texte)
    if not correspondances:
        return "[invalid]"
    lettre = correspondances[-1].upper()
    return _MAPPING.get(lettre, "[invalid]")


def _echantillons():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_chargee():
    ech = _echantillons()
    assert len(ech) == 20, f"la fixture n'a plus 20 échantillons réels : {len(ech)}"


def test_rejoue_fidelement_les_20_conclusions_reelles():
    ech = _echantillons()
    divergences = []
    for i, e in enumerate(ech):
        obtenu = get_answer_v275(e["resp"])
        attendu = e["filtered"][0] if isinstance(e["filtered"], list) else e["filtered"]
        if obtenu != attendu:
            divergences.append((i, attendu, obtenu, e["resp"][:120]))
    assert not divergences, f"{len(divergences)}/20 divergent de l'extraction réelle de lm-eval : {divergences[:3]}"


def test_detecte_une_conclusion_sans_mot_answer():
    assert get_answer_v275("Je pense que c'est (C) mais je ne conclus pas avec le mot magique.") == "[invalid]"


def test_prend_la_derniere_occurrence():
    # un modèle qui hésite puis conclut : la DERNIÈRE mention de "answer" compte (group_select -1)
    assert get_answer_v275("First I thought answer was (A), but the final answer is (D).") == "(D)"
