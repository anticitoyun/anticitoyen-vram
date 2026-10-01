"""e50.3 § 7 (chef, 01/10, après chargement réel) : `gsm8k_e50`/`humaneval_e50` levaient
`HfUriError` au chargement (« Repository id must be 'namespace/name', got 'gsm8k' ») — le
`huggingface_hub` installé dans le venv lm-eval réel exige un id complet. Corrigé en
`openai/gsm8k` et `openai/openai_humaneval`.

Ce test charge RÉELLEMENT les 5 tâches e50 avec un vrai lm-eval installé :
`TaskManager(include_path=...).load_task_or_group`, `eval_docs`, `doc_to_text` du premier
document. Aucune carte, aucun modèle, aucun serveur — juste le chargement du dataset et le
rendu du gabarit de prompt.

Interpréteur résolu par le MÊME ordre que `outils/qualite-e50.sh` (chef, 01/10, après
fusion) : `$ACVRAM_LMEVAL_PY`, puis `$DEPOT/.venv-panel/bin/python` (l'emplacement RÉEL que
`qualite-e50.sh`/`panel-taches.sh` utilisent) en premier recours sérieux, puis la copie figée
TEMPORAIRE de la campagne 275 (`travail/poste2-275-figee/.venv-panel`) en tout dernier — dès
qu'un `.venv-panel` réel existe sous ce worktree, ce test (et `qualite-e50.sh`) le préfère
automatiquement et ne dépend plus de la copie figée. Saute proprement si aucun des trois
n'est présent."""
import json
import os
import shutil
import subprocess
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resoudre_py_lmeval():
    """Même ordre que `outils/qualite-e50.sh` : $ACVRAM_LMEVAL_PY, $DEPOT/.venv-panel,
    repli temporaire (copie figée 275) — rend None si aucun n'est un exécutable."""
    candidats = [
        os.environ.get("ACVRAM_LMEVAL_PY"),
        os.path.join(ICI, ".venv-panel", "bin", "python"),
        os.path.expanduser("~/Bureau/Claude/travail/poste2-275-figee/.venv-panel/bin/python"),
    ]
    for c in candidats:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


PY_LMEVAL = _resoudre_py_lmeval()

pytestmark = pytest.mark.skipif(
    not PY_LMEVAL or not shutil.which("bash"),
    reason="aucun interprète lm-eval trouvé (ACVRAM_LMEVAL_PY, .venv-panel, ou copie figée 275)")

TACHES = ["mmlu_e50_hsm", "mmlu_e50_law", "mmlu_e50_ccs", "gsm8k_e50", "humaneval_e50"]

SCRIPT = r"""
import sys, json, warnings
warnings.filterwarnings("ignore")
from lm_eval.tasks import TaskManager
tm = TaskManager(include_path=%(include_path)r)
out = {}
for nom in %(taches)r:
    try:
        d = tm.load_task_or_group(nom)
        t = d[nom] if isinstance(d, dict) else d
        docs = list(t.eval_docs)
        doc = docs[0]
        texte = t.doc_to_text(doc)
        out[nom] = {"ok": True, "n_docs": len(docs), "doc_to_text_0": str(texte)[:300]}
    except Exception as e:
        out[nom] = {"ok": False, "erreur": f"{type(e).__name__}: {e}"[:400]}
print("RESULTAT_E50_TACHES " + json.dumps(out))
"""


def _charger_les_taches():
    include_path = os.path.join(ICI, "outils", "lm_eval_taches")
    code = SCRIPT % {"include_path": include_path, "taches": TACHES}
    r = subprocess.run([PY_LMEVAL, "-c", code], capture_output=True, text=True, timeout=120)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("RESULTAT_E50_TACHES ")]
    assert lignes, f"rien imprimé : rc={r.returncode}\nstdout={r.stdout[-2000:]}\nstderr={r.stderr[-2000:]}"
    return json.loads(lignes[-1][len("RESULTAT_E50_TACHES "):])


def test_les_cinq_taches_se_chargent_avec_le_vrai_lm_eval():
    resultats = _charger_les_taches()
    echecs = {nom: r["erreur"] for nom, r in resultats.items() if not r["ok"]}
    assert not echecs, f"tâche(s) qui ne se chargent pas : {echecs}"


def test_chaque_tache_a_des_documents_et_un_doc_to_text_non_vide():
    resultats = _charger_les_taches()
    for nom in TACHES:
        r = resultats[nom]
        assert r["ok"], r.get("erreur")
        assert r["n_docs"] > 0, f"{nom} : 0 document"
        assert r["doc_to_text_0"].strip(), f"{nom} : doc_to_text vide sur le 1er document"


def test_gsm8k_e50_et_humaneval_e50_cassent_sur_lancien_dataset_path(tmp_path):
    """Cassant : rejoue le chargement avec `dataset_path: gsm8k` / `openai_humaneval` (l'état
    d'avant ce correctif, chef 01/10) dans une copie isolée des yaml — doit ÉCHOUER."""
    dossier = tmp_path / "lm_eval_taches"
    dossier.mkdir()
    for nom in ("gsm8k_e50", "humaneval_e50"):
        source = os.path.join(ICI, "outils", "lm_eval_taches", f"{nom}.yaml")
        texte = open(source, encoding="utf-8").read()
        texte = texte.replace("dataset_path: openai/gsm8k", "dataset_path: gsm8k")
        texte = texte.replace("dataset_path: openai/openai_humaneval", "dataset_path: openai_humaneval")
        (dossier / f"{nom}.yaml").write_text(texte)
    code = SCRIPT % {"include_path": str(dossier), "taches": ["gsm8k_e50", "humaneval_e50"]}
    r = subprocess.run([PY_LMEVAL, "-c", code], capture_output=True, text=True, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("RESULTAT_E50_TACHES ")]
    assert lignes, r.stdout[-2000:] + r.stderr[-2000:]
    resultats = json.loads(lignes[-1][len("RESULTAT_E50_TACHES "):])
    echoues = {nom: r for nom, r in resultats.items() if not r["ok"]}
    assert echoues, (
        "l'ancien dataset_path (gsm8k/openai_humaneval, sans namespace) se charge quand même — "
        "ce test ne distinguerait plus l'ancien du nouveau")
    for nom, r in echoues.items():
        assert "HfUriError" in r["erreur"] or "namespace" in r["erreur"].lower() or "Repository id" in r["erreur"], (
            f"{nom} échoue mais pas pour la raison attendue : {r['erreur']}")
