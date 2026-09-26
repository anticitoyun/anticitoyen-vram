"""Pièce 266 b : le job flatpak de release.yml prend packaging/flathub/ (outillage de construction : manifeste, sources-torch.sh,
roue_url.py) de la REF DU WORKFLOW (github.sha), la source d'acvram restant au tag. Sans cela, une relance de la release
v0.7.0 reconstruirait avec le sources-torch.sh cassé du tag et le correctif de main ne servirait qu'à la release suivante."""
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"


def _flatpak(texte: str | None = None) -> dict:
    return yaml.safe_load((texte or RELEASE.read_text(encoding="utf-8")))["jobs"]["flatpak"]


def outillage_depuis_la_ref_du_workflow(job: dict) -> bool:
    """Vrai si : checkout au tag avec fetch-depth 0, puis, AVANT toute étape qui lit packaging/flathub, un
    `git checkout ${{ github.sha }} -- packaging/flathub`."""
    etapes = job["steps"]
    i_co = next((i for i, s in enumerate(etapes) if str(s.get("uses", "")).startswith("actions/checkout")), None)
    if i_co is None:
        return False
    with_ = etapes[i_co].get("with") or {}
    if "env.TAG" not in str(with_.get("ref", "")) or str(with_.get("fetch-depth", "")) != "0":
        return False
    motif = re.compile(r"git checkout \$\{\{\s*github\.sha\s*\}\} -- packaging/flathub\b")
    i_out = next((i for i, s in enumerate(etapes) if motif.search(s.get("run", ""))), None)
    if i_out is None or i_out <= i_co:
        return False
    premiere_lecture = next((i for i, s in enumerate(etapes) if "packaging/flathub" in s.get("run", "") and i != i_out), None)
    return premiere_lecture is None or i_out < premiere_lecture


def test_le_job_flatpak_prend_l_outillage_de_la_ref_du_workflow():
    assert outillage_depuis_la_ref_du_workflow(_flatpak()), \
        "flatpak : checkout au tag (fetch-depth 0) puis `git checkout ${{ github.sha }} -- packaging/flathub` avant toute lecture"


def test_la_source_d_acvram_reste_au_tag():
    co = next(s for s in _flatpak()["steps"] if str(s.get("uses", "")).startswith("actions/checkout"))
    assert "env.TAG" in str((co.get("with") or {}).get("ref", "")), "flatpak : la source d'acvram doit rester au tag"


def test_le_test_sait_dire_faux():
    job = _flatpak()
    sans = {"steps": [s for s in job["steps"] if "github.sha" not in s.get("run", "")]}
    assert not outillage_depuis_la_ref_du_workflow(sans), "sans l'extraction depuis github.sha, la garde doit être rouge"
    peu_profond = {"steps": [dict(s, **{"with": {"ref": "${{ env.TAG }}"}}) if "checkout" in str(s.get("uses", "")) else s for s in job["steps"]]}
    assert not outillage_depuis_la_ref_du_workflow(peu_profond), "sans fetch-depth 0, github.sha n'est pas extractible : rouge"
