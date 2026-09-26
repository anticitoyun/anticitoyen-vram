"""Pièce 266 b : le job flatpak de release.yml prend packaging/flathub/ (outillage de construction : manifeste, sources-torch.sh,
roue_url.py) de la REF DU WORKFLOW (github.sha), la source d'acvram restant au tag. Sans cela, une relance de la release
v0.7.0 reconstruirait avec le sources-torch.sh cassé du tag et le correctif de main ne servirait qu'à la release suivante."""
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"


def _jobs(texte: str | None = None) -> dict:
    return yaml.safe_load((texte or RELEASE.read_text(encoding="utf-8")))["jobs"]


def _flatpak(texte: str | None = None) -> dict:
    return _jobs(texte)["flatpak"]


def jobs_qui_lisent_packaging_flathub(jobs: dict) -> list[str]:
    """266 f : tout job dont une étape `run` lit packaging/flathub/ (PYTHON_RUNTIME, sources-*.py, manifeste)."""
    return [nom for nom, j in jobs.items()
            if any("packaging/flathub" in s.get("run", "") and "git checkout" not in s.get("run", "") for s in j.get("steps", []))]


def outillage_depuis_la_ref_du_workflow(job: dict) -> bool:
    """Vrai si la source est au tag (checkout `ref: env.TAG`) et que, AVANT toute étape qui lit packaging/flathub, ce dossier
    vient de la ref du workflow par l'un des deux mécanismes :
    (a) checkout au tag avec fetch-depth 0 puis `git checkout ${{ github.sha }} -- packaging/flathub` (job flatpak, runner avec git) ;
    (b) second `actions/checkout` avec `ref: ${{ github.sha }}` et `path: <dossier>`, puis une étape qui copie
        `<dossier>/packaging/flathub` vers `packaging/flathub` (266 f : conteneurs sans git, où le checkout n'a pas de .git)."""
    etapes = job["steps"]
    i_co = next((i for i, s in enumerate(etapes) if str(s.get("uses", "")).startswith("actions/checkout")
                 and "env.TAG" in str((s.get("with") or {}).get("ref", ""))), None)
    if i_co is None:
        return False
    i_out = None
    with_ = etapes[i_co].get("with") or {}
    if str(with_.get("fetch-depth", "")) == "0":
        motif = re.compile(r"git checkout \$\{\{\s*github\.sha\s*\}\} -- packaging/flathub\b")
        i_out = next((i for i, s in enumerate(etapes) if motif.search(s.get("run", ""))), None)
    if i_out is None:
        i_co2 = next((i for i, s in enumerate(etapes) if i > i_co and str(s.get("uses", "")).startswith("actions/checkout")
                      and "github.sha" in str((s.get("with") or {}).get("ref", "")) and (s.get("with") or {}).get("path")), None)
        if i_co2 is not None:
            chemin = str(etapes[i_co2]["with"]["path"]).rstrip("/")
            motif = re.compile(rf"cp -a {re.escape(chemin)}/packaging/flathub packaging/flathub\b")
            i_out = next((i for i, s in enumerate(etapes) if i > i_co2 and motif.search(s.get("run", ""))), None)
    if i_out is None or i_out <= i_co:
        return False
    premiere_lecture = next((i for i, s in enumerate(etapes)
                             if "packaging/flathub" in s.get("run", "") and i != i_out and "cp -a" not in s.get("run", "")), None)
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


def test_tout_job_qui_lit_packaging_flathub_le_prend_de_la_ref_du_workflow():
    """266 f : noyaux-precompiles lisait PYTHON_RUNTIME au tag v0.7.0, où il n'existe pas (run 36227010415) — la même
    extraction que le job flatpak (266 b) s'impose à tout job qui lit ce dossier d'outillage."""
    jobs = _jobs()
    lecteurs = jobs_qui_lisent_packaging_flathub(jobs)
    assert set(lecteurs) >= {"flatpak", "noyaux-precompiles"}, lecteurs      # gabarit : les deux lecteurs d'aujourd'hui
    fautifs = [n for n in lecteurs if not outillage_depuis_la_ref_du_workflow(jobs[n])]
    assert not fautifs, f"jobs qui lisent packaging/flathub sans l'extraire de github.sha : {fautifs}"


def test_le_temoin_266f():
    jobs = _jobs()
    sans = {"steps": [s for s in jobs["noyaux-precompiles"]["steps"]
                      if "github.sha" not in str((s.get("with") or {}).get("ref", "")) and "cp -a" not in s.get("run", "")]}
    assert "noyaux-precompiles" in jobs_qui_lisent_packaging_flathub(jobs)
    assert not outillage_depuis_la_ref_du_workflow(sans), "le job noyaux de la v0.7.0 relancée (sans extraction) doit être rouge"
