"""Pièce 259 b : release.yml publie SHA256SUMS après TOUS les jobs qui joignent un fichier à la release. Casse si un
job qui fait `gh release upload` n'est pas dans `needs` du job `sommes` (ses fichiers ne seraient pas couverts), si
`sommes` ne télécharge pas la release avant de sommer, ou s'il ne joint pas SHA256SUMS."""
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"


def _jobs() -> dict:
    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]


def _run(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job.get("steps", []))


def _jobs_qui_joignent(jobs: dict) -> set:
    return {nom for nom, j in jobs.items() if nom != "sommes" and re.search(r"gh release upload\b", _run(j))}


def test_tout_job_qui_joint_est_couvert_par_sommes():
    jobs = _jobs()
    assert "sommes" in jobs, "release.yml : job `sommes` absent"
    joignent = _jobs_qui_joignent(jobs)
    assert joignent >= {"deb", "rpm", "aur", "flatpak", "translations"}, joignent   # gabarit : ceux d'aujourd'hui
    needs = set(jobs["sommes"].get("needs") or [])
    assert joignent <= needs, f"jobs qui joignent un fichier hors de needs de `sommes` : {sorted(joignent - needs)}"


def test_sommes_telecharge_puis_joint_sha256sums():
    run = _run(_jobs()["sommes"])
    i_dl, i_sum, i_up = run.find("gh release download"), run.find("sha256sum"), run.find("gh release upload")
    assert 0 <= i_dl < i_sum < i_up, "sommes : télécharger la release, sommer, puis joindre — dans cet ordre"
    assert "SHA256SUMS" in run[i_up:], "sommes : le fichier joint doit s'appeler SHA256SUMS (lu par verifier-release.sh)"
    assert "rm -f SHA256SUMS" in run, "sommes : un SHA256SUMS d'un passage précédent ne doit pas entrer dans les sommes"
    assert "!cancelled()" in str(_jobs()["sommes"].get("if", "")), "sommes : tourne même si un job a échoué (fichiers présents sommés)"


def test_le_test_sait_dire_faux():
    """Un job qui joint et que `sommes` n'attend pas doit être vu."""
    jobs = _jobs()
    jobs["fantome"] = {"steps": [{"run": 'gh release upload "$TAG" fantome.bin'}]}
    assert not _jobs_qui_joignent(jobs) <= set(jobs["sommes"]["needs"])
