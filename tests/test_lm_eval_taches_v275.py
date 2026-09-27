"""Pièce 275d (poste2, 26/09) : le correctif d'extraction (275b/c) vit DANS le banc — une
config lm-eval dérivée (`outils/lm_eval_taches/mmlu_v275_*.yaml`), pas un post-traitement
(sinon la prochaine mesure retombe dans le piège de la 275). Deux niveaux de vérification,
SANS GPU, SANS nouvelle génération :

1. la config est VALIDE pour lm-eval (`lm_eval validate`, sous-processus, .venv-panel) ;
2. la VRAIE chaîne de filtres qu'elle déclare (chargée depuis le fichier, pas réécrite ici)
   note correctement les 30 sorties déjà écrites et classées à la main (275b/c).

Saute si `.venv-panel` (lm-eval) ou la référence 275 (hors git) sont absents de ce poste.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
VENV_PANEL = RACINE / ".venv-panel" / "bin" / "python"
LM_EVAL = RACINE / ".venv-panel" / "bin" / "lm_eval"
TACHES_DIR = RACINE / "outils" / "lm_eval_taches"
REF_MIXTE = Path.home() / ".cache" / "acvram" / "qualite-275" / "Qwen3.8-27B-unsloth-mixte-i8c"

pytestmark = pytest.mark.skipif(not VENV_PANEL.exists(), reason="lm-eval (.venv-panel) non installé sur ce poste")

_TACHES_V275 = [
    "mmlu_v275_high_school_mathematics",
    "mmlu_v275_professional_law",
    "mmlu_v275_college_computer_science",
]
_ANCIEN_NOM = {
    "mmlu_v275_high_school_mathematics": "mmlu_flan_cot_fewshot_high_school_mathematics",
    "mmlu_v275_professional_law": "mmlu_flan_cot_fewshot_professional_law",
    "mmlu_v275_college_computer_science": "mmlu_flan_cot_fewshot_college_computer_science",
}
# scores attendus (275c, comptés sur les 150/150/100 échantillons du mixte-i8c) : plancher
# large pour ne pas casser sur un arrondi, la vraie valeur est bien au-dessus.
_SCORE_MIN = {
    "mmlu_v275_high_school_mathematics": 0.85,
    "mmlu_v275_professional_law": 0.60,
    "mmlu_v275_college_computer_science": 0.75,
}


def test_les_configs_v275_sont_valides_pour_lm_eval():
    r = subprocess.run(
        [str(LM_EVAL), "validate", "--tasks", ",".join(_TACHES_V275), "--include_path", str(TACHES_DIR)],
        capture_output=True, text=True)
    assert r.returncode == 0, f"config invalide :\n{r.stdout}\n{r.stderr}"
    assert "All tasks found and valid" in r.stdout


@pytest.mark.skipif(not REF_MIXTE.exists(), reason="référence 275 (mixte-i8c, hors git) absente de ce poste")
@pytest.mark.parametrize("tache_v275", _TACHES_V275)
def test_la_chaine_de_filtres_chargee_note_juste(tache_v275):
    """Charge la VRAIE chaîne de filtres depuis le yaml (pas une copie) et l'applique aux
    sorties déjà écrites — preuve que le fichier de config, pas une réimplémentation, corrige
    le score."""
    sys.path.insert(0, str(RACINE / ".venv-panel" / "lib" / "python3.12" / "site-packages"))
    import yaml
    from lm_eval.filters.extraction import RegexFilter
    from lm_eval.filters.transformation import MapFilter, UppercaseFilter

    cfg = yaml.safe_load(open(TACHES_DIR / f"{tache_v275}.yaml", encoding="utf-8"))
    chaine = []
    for etape in cfg["filter_list"][0]["filter"]:
        etape = dict(etape)
        fn = etape.pop("function")
        if fn == "regex":
            chaine.append(RegexFilter(**etape))
        elif fn == "uppercase":
            chaine.append(UppercaseFilter())
        elif fn == "map":
            chaine.append(MapFilter(**etape))
        else:
            raise AssertionError(f"fonction de filtre inattendue dans la config : {fn}")

    ancien_nom = _ANCIEN_NOM[tache_v275]
    fichiers = glob.glob(str(REF_MIXTE / f"{ancien_nom}.json.echantillons" / "**" / f"samples_{ancien_nom}_*.jsonl"),
                          recursive=True)
    assert fichiers, f"aucun échantillon pour {ancien_nom} sous {REF_MIXTE}"
    lignes = [json.loads(l) for l in open(fichiers[0], encoding="utf-8")]
    lignes = [l for l in lignes if l.get("filter") == "get-answer"]
    assert lignes, "aucune ligne avec filter=get-answer (ancien filtre) dans l'échantillon"

    resps = [[l["resps"][0][0]] for l in lignes]
    docs = [{} for _ in resps]
    for f in chaine:
        resps = f.apply(resps, docs)
    sorties = [r[0] for r in resps]
    cibles = [l["target"] for l in lignes]

    score = sum(1 for s, c in zip(sorties, cibles) if s == c) / len(lignes)
    assert score >= _SCORE_MIN[tache_v275], (
        f"{tache_v275} : score {score:.4f} < plancher {_SCORE_MIN[tache_v275]} — "
        f"la config chargée ne corrige pas le score comme prévu (275b/c)")
