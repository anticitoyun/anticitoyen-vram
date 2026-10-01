"""Pièce 275 (poste2, ordre chef, 30/09) : verifier-reference-275.sh doit REFUSER
nommément une référence qualite-275 au format ANCIEN (échantillons sous les noms de
tâche d'avant le renommage de la pièce 275d, mmlu_flan_cot_fewshot_* au lieu de
mmlu_v275_*), sans avoir besoin de carte GPU — trouvé le 30/09 : les références
Coder-30B et mixte-27B (générées 26/09) plantaient qualite.sh en cours de route (un
`find` sans message clair) APRÈS avoir déjà dépensé une prise carte.sh.
"""
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
SCRIPT = RACINE / "outils" / "verifier-reference-275.sh"

TACHES = [
    "gsm8k",
    "mmlu_v275_high_school_mathematics",
    "mmlu_v275_professional_law",
    "mmlu_v275_college_computer_science",
]


def _lancer(home):
    return subprocess.run(
        [str(SCRIPT), "un-modele"], cwd=RACINE, env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )


def _ref(home):
    ref = home / ".cache" / "acvram" / "qualite-275" / "un-modele"
    ref.mkdir(parents=True)
    (ref / "ppl.json").write_text("{}")
    return ref


def _ecrire_echantillon(ref, tache_fichier, tache_nom):
    d = ref / f"{tache_fichier}.json.echantillons" / "coder"
    d.mkdir(parents=True)
    (d / f"samples_{tache_nom}_2026-09-30T00-00-00.000000.jsonl").write_text("{}\n")


def test_refus_sans_reference(tmp_path):
    r = _lancer(tmp_path)
    assert r.returncode == 66
    assert "REFUS" in r.stdout


def test_refus_ancien_format(tmp_path):
    ref = _ref(tmp_path)
    # Format ANCIEN : panel.json.echantillons, noms de tâche d'avant 275d — reproduit
    # exactement la structure trouvée sur Coder-30B/mixte-27B le 30/09.
    d = ref / "panel.json.echantillons" / "mmlu" / "coder"
    d.mkdir(parents=True)
    (d / "samples_mmlu_flan_cot_fewshot_professional_law_2026-09-26T00-00-00.jsonl").write_text("{}\n")
    d2 = ref / "panel.json.echantillons" / "gsm8k" / "coder"
    d2.mkdir(parents=True)
    (d2 / "samples_gsm8k_2026-09-26T00-00-00.jsonl").write_text("{}\n")

    r = _lancer(tmp_path)
    assert r.returncode == 72
    assert "REFUS" in r.stdout
    assert "ANCIEN" in r.stdout


def test_ok_format_courant(tmp_path):
    ref = _ref(tmp_path)
    for tache in TACHES:
        _ecrire_echantillon(ref, tache, tache)

    r = _lancer(tmp_path)
    assert r.returncode == 0
    assert "OK" in r.stdout


def test_refus_format_partiel(tmp_path):
    """3 tâches migrées, 1 encore à l'ancien format — doit refuser, pas se satisfaire
    de la majorité (une seule tâche mal formée suffit à fausser McNemar dessus)."""
    ref = _ref(tmp_path)
    for tache in TACHES[:3]:
        _ecrire_echantillon(ref, tache, tache)
    d = ref / f"{TACHES[3]}.json.echantillons" / "coder"
    d.mkdir(parents=True)
    (d / "samples_mmlu_flan_cot_fewshot_college_computer_science_2026-09-26T00-00-00.jsonl").write_text("{}\n")

    r = _lancer(tmp_path)
    assert r.returncode == 72
