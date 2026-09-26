"""Pièce 281 (chef, 26/09) : `outils/sortir-version.sh` codifie la sortie de version faite
à la main — refus net (message + code) sur chaque précondition manquante, `--simule` trace
tout sans rien pousser ni créer de tag. Dépôt jetable, jamais le vrai (aucun push, aucun gh,
aucun jeton lu dans ces épreuves — les refus précèdent tous ces gestes)."""
import pathlib
import shutil
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "sortir-version.sh"
CARTE = RACINE / "outils" / "carte.sh"
V = "0.7.3"
VNUM = "0.7.3"


def _depot_jetable(tmp_path: pathlib.Path, *, version: str = VNUM, avec_notes: bool = True) -> pathlib.Path:
    d = tmp_path / "depot"
    d.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.name", "t"], check=True)
    (d / "pyproject.toml").write_text(f'[project]\nname = "acvram"\nversion = "{version}"\n', encoding="utf-8")
    outils = d / "outils"; outils.mkdir()
    shutil.copy2(SCRIPT, outils / "sortir-version.sh")
    (outils / "sortir-version.sh").chmod(0o755)
    shutil.copy2(CARTE, outils / "carte.sh")
    (outils / "carte.sh").chmod(0o755)
    # publier-github.sh et verifier-release.sh : jamais réellement appelés dans ces épreuves
    # (les refus précèdent, ou --simule les trace sans les exécuter) — pas besoin de vrais fichiers.
    if avec_notes:
        notes = d / "docs" / "notes"; notes.mkdir(parents=True)
        (notes / f"release-v{version}-github.md").write_text("notes", encoding="utf-8")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-m", "initial"], check=True)
    return d


def _lancer(depot: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(depot / "outils" / "sortir-version.sh"), *args],
                          cwd=depot, capture_output=True, text=True, timeout=60)


def test_refuse_si_un_fichier_suivi_est_modifie(tmp_path):
    d = _depot_jetable(tmp_path)
    (d / "pyproject.toml").write_text(f'[project]\nname = "acvram"\nversion = "{VNUM}"\n# modifié\n', encoding="utf-8")
    r = _lancer(d, f"v{V}")
    assert r.returncode == 65 and "n'est pas propre" in r.stderr, r.stderr


def test_accepte_un_fichier_non_suivi_sur_main(tmp_path):
    """282 (chef) : un scratchpad/<piece>/ non suivi ne bloque pas — seul le contenu
    SUIVI compte, comme sur la vraie main pendant une pièce en cours."""
    d = _depot_jetable(tmp_path)
    (d / "scratchpad-en-cours.txt").write_text("x", encoding="utf-8")
    r = _lancer(d, f"v{V}", "--simule")
    assert r.returncode == 0, r.stdout + r.stderr


def test_refuse_si_version_pyproject_differente(tmp_path):
    d = _depot_jetable(tmp_path, version="0.7.2")
    r = _lancer(d, f"v{V}")
    assert r.returncode == 65 and "pyproject.toml porte 0.7.2" in r.stderr, r.stderr


def test_refuse_si_notes_github_absentes(tmp_path):
    d = _depot_jetable(tmp_path, avec_notes=False)
    r = _lancer(d, f"v{V}")
    assert r.returncode == 66 and f"release-v{V}-github.md introuvable" in r.stderr, r.stderr


def test_refuse_si_le_tag_existe_deja(tmp_path):
    d = _depot_jetable(tmp_path)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "deja"], check=True)
    r = _lancer(d, f"v{V}", "--simule")
    assert r.returncode == 67 and "existe déjà" in r.stderr, r.stderr


def test_refuse_sur_une_version_malformee(tmp_path):
    d = _depot_jetable(tmp_path)
    r = _lancer(d, "0.7.3")   # sans le "v"
    assert r.returncode == 64, r.stderr


def test_simule_trace_tout_sans_creer_de_tag_ni_pousser(tmp_path):
    d = _depot_jetable(tmp_path)
    r = _lancer(d, f"v{V}", "--simule")
    assert r.returncode == 0, r.stdout + r.stderr
    for attendu in ("[simulé] git tag -a", "[simulé] git push origin", "publier-github.sh",
                    "gh release create", "gh run watch", "verifier-release.sh"):
        assert attendu in r.stdout, (attendu, r.stdout)
    # rien de réel n'a bougé : aucun tag, dépôt toujours à un seul commit
    tags = subprocess.run(["git", "-C", str(d), "tag"], capture_output=True, text=True, check=True).stdout
    assert tags.strip() == "", f"--simule a quand même créé un tag : {tags!r}"


def test_le_jeton_n_apparait_jamais_dans_la_trace_simulee(tmp_path):
    """Témoin : si un jour la trace --simule imprimait la commande gh avec un jeton en
    clair, ce test doit le voir — aucun jeton n'est lu ici (refus impossible), donc rien
    qui ressemble à une valeur de jeton ne doit apparaître, seulement le nom de la commande."""
    d = _depot_jetable(tmp_path)
    r = _lancer(d, f"v{V}", "--simule")
    assert "GH_TOKEN=" not in r.stdout and "ghp_" not in r.stdout and "github_pat_" not in r.stdout
