"""e50.3 § 7 (poste3, 01/10) : les 5 yaml de la batterie e50 sont du YAML valide, portent un
`task:` nommé (sans ça lm-eval ne peut pas les charger par `--include_path`), 0-shot, et le
`max_gen_toks` figé par la méthode (768 MMLU/GSM8K, 512 HumanEval, § 1-2). Ce test casse si un
yaml change de limite/fewshot SANS que son nom change (REGLES § 1, régime porté par le nom) —
il n'exécute aucun lm-eval."""
import glob
import os
import yaml

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER = os.path.join(ICI, "outils", "lm_eval_taches")

# pin : NOM -> (num_fewshot, max_gen_toks) — un yaml e50 qui change l'un des deux sans
# renommer la tâche casse ici, avant d'atteindre une vraie campagne.
ATTENDU = {
    "mmlu_e50_hsm": (0, 768),
    "mmlu_e50_law": (0, 768),
    "mmlu_e50_ccs": (0, 768),
    "gsm8k_e50": (0, 768),
    "humaneval_e50": (0, 512),
}


def _charger(nom):
    with open(os.path.join(DOSSIER, f"{nom}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_les_cinq_yaml_existent_et_se_parsent():
    presents = {os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(DOSSIER, "*_e50*.yaml")) + glob.glob(os.path.join(DOSSIER, "*e50_*.yaml"))}
    manquants = set(ATTENDU) - presents
    assert not manquants, f"yaml e50 manquant(s) : {manquants}"


def test_chaque_yaml_porte_son_nom_de_tache():
    for nom in ATTENDU:
        d = _charger(nom)
        assert d.get("task") == nom, f"{nom}.yaml : task={d.get('task')!r}, attendu {nom!r}"


def test_fewshot_et_max_gen_toks_pin():
    divergences = []
    for nom, (fewshot_attendu, toks_attendu) in ATTENDU.items():
        d = _charger(nom)
        fewshot = d.get("num_fewshot")
        toks = (d.get("generation_kwargs") or {}).get("max_gen_toks")
        if (fewshot, toks) != (fewshot_attendu, toks_attendu):
            divergences.append((nom, (fewshot, toks), (fewshot_attendu, toks_attendu)))
    assert not divergences, (
        f"num_fewshot/max_gen_toks changés sans renommer la tâche : {divergences} — "
        "REGLES § 1, le régime est porté par le nom")


def test_mmlu_e50_filtre_identique_au_275d():
    """Les 3 MMLU e50 partagent le filtre `get-answer-v275` (voir test_qualite_e50_extraction.py,
    validé sur 20 conclusions réelles) — jamais un filtre maison non éprouvé."""
    for nom in ("mmlu_e50_hsm", "mmlu_e50_law", "mmlu_e50_ccs"):
        d = _charger(nom)
        noms_filtres = [f["name"] for f in d.get("filter_list", [])]
        assert "get-answer-v275" in noms_filtres, f"{nom}.yaml : filtre get-answer-v275 absent"


def test_humaneval_e50_ne_porte_aucune_metrique_executrice():
    """§ 2 bis : humaneval_e50 ne doit JAMAIS exécuter de code via lm-eval lui-même
    (`metric_list` natif comme `pass_at_k` suppose `confirm_run_unsafe_code`) — le score vient
    uniquement de `outils/qualite-e50-humaneval-score.py` sous bac à sable."""
    d = _charger("humaneval_e50")
    assert d.get("metric_list") == [], (
        "humaneval_e50.yaml porte une métrique native — risque d'exécution non sandboxée, "
        "REGLES § 6")
