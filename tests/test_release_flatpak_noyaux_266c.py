"""Pièce 266 c : dans le job flatpak de release.yml, le chemin des noyaux précompilés que lit le manifeste (module
acvram-noyaux, après la réécriture `sed` du job) est EXACTEMENT le `path:` de download-artifact — et le répertoire de
construction de flatpak-builder (que `--force-clean` efface) n'est ni ce chemin ni un de ses parents : la v0.7.0 relancée
construisait dans `build`, effaçait `build/noyaux` et ne trouvait plus sa source."""
import pathlib
import re

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"
MANIFESTE = RACINE / "packaging" / "flathub" / "io.github.anticitoyen.acvram.yml"


def _job(texte: str | None = None) -> dict:
    return yaml.safe_load(texte or RELEASE.read_text(encoding="utf-8"))["jobs"]["flatpak"]


def _run(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in job.get("steps", []))


def chemin_artefact(job: dict) -> str:
    return next(str((s.get("with") or {})["path"]) for s in job["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact"))


def chemin_manifeste_apres_sed(job: dict, manifeste: str) -> str:
    """Le `path:` du module acvram-noyaux tel que le job le réécrit (les `sed -e 's#…#…#'` de la construction)."""
    m = yaml.safe_load(manifeste)
    module = next(x for x in m["modules"] if isinstance(x, dict) and x.get("name") == "acvram-noyaux")
    chemin = next(s["path"] for s in module["sources"] if s.get("type") == "dir")
    ligne = f"path: {chemin}"
    run = re.sub(r"\\\n\s*", " ", _run(job))
    for motif, rempl in re.findall(r"-e 's#([^#]*)#([^#]*)#'", run):
        ligne = re.sub(motif, rempl, ligne)
    return ligne.split("path: ", 1)[1].strip()


def repertoire_de_construction(job: dict) -> str:
    """Premier argument positionnel de flatpak-builder (le répertoire de construction), avant le manifeste."""
    run = re.sub(r"\\\n\s*", " ", _run(job))
    cmd = next(l for l in run.splitlines() if l.strip().startswith("flatpak-builder "))
    args = [a for a in cmd.split()[1:] if not a.startswith("-")]
    return args[0]


def test_le_manifeste_lit_les_noyaux_la_ou_download_artifact_les_pose():
    job = _job()
    assert chemin_manifeste_apres_sed(job, MANIFESTE.read_text(encoding="utf-8")) == chemin_artefact(job)


def test_le_repertoire_de_construction_n_efface_pas_les_noyaux():
    job = _job()
    construction, noyaux = repertoire_de_construction(job), chemin_artefact(job)
    assert construction != noyaux and not noyaux.startswith(construction.rstrip("/") + "/"), \
        f"flatpak-builder --force-clean efface `{construction}`, qui contient `{noyaux}`"


def test_le_test_sait_dire_faux():
    """Le job de la v0.7.0 relancée (répertoire `build`) doit être vu."""
    ancien = RELEASE.read_text(encoding="utf-8").replace("--repo=repo construction-flatpak io.github", "--repo=repo build io.github")
    job = _job(ancien)
    assert repertoire_de_construction(job) == "build" and chemin_artefact(job).startswith("build/")
