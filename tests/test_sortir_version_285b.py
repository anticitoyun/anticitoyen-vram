"""Pièce 285 b (chef, sur la vraie sortie 0.7.4) : deux défauts trouvés en conditions
réelles.
(a) L'attente du run « Release packages » prenait `head -1` sans filtrer sur le tag — un
    run release-triggered porte `headBranch` = le TAG (pas main), donc un `head -1` sans
    filtre peut surveiller le run d'une AUTRE sortie. `gh` simulé (faux binaire dans le
    PATH du test) : deux tags/runs, seul celui du bon tag doit être surveillé.
(b) La CI publique a échoué à l'étape 4 alors que le tag était déjà posé — une relance
    sans reprise tombe sur le code 67 (« tag existe déjà »). `--depuis N` (4..7) saute les
    étapes 1-3, mais REFUSE (72) si le tag n'existe pas ou n'est pas ancêtre de main — pour
    ne jamais reprendre sur le tag d'une autre sortie."""
import os
import pathlib
import shutil
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = RACINE / "outils" / "sortir-version.sh"
CARTE = RACINE / "outils" / "carte.sh"
VERIFIE = RACINE / "outils" / "verifier-release.sh"
V = "0.7.4"
VNUM = "0.7.4"


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
    shutil.copy2(VERIFIE, outils / "verifier-release.sh")
    (outils / "verifier-release.sh").chmod(0o755)
    if avec_notes:
        notes = d / "docs" / "notes"; notes.mkdir(parents=True)
        (notes / f"release-v{version}-github.md").write_text("notes", encoding="utf-8")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-m", "initial"], check=True)
    return d


def _lancer(depot: pathlib.Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    full_env = dict(os.environ, **env)
    return subprocess.run(["bash", str(depot / "outils" / "sortir-version.sh"), *args],
                          cwd=depot, capture_output=True, text=True, timeout=60, env=full_env)


@pytest.fixture
def faux_home():
    """verifier-release.sh refuse un dossier par défaut sous /tmp (259) — $HOME simulé hors /tmp."""
    base = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "acvram" / "tests-285b"
    d = base / f"p{os.getpid()}"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _faux_gh(dossier: pathlib.Path, script: str) -> dict:
    """Un `gh` bidon dans le PATH — jamais de réseau, jamais de vrai jeton lu."""
    (dossier / "gh").write_text(f"#!/usr/bin/env bash\n{script}\n", encoding="utf-8")
    (dossier / "gh").chmod(0o755)
    return {"PATH": f"{dossier}:{os.environ['PATH']}"}


# ---- (a) le run doit être filtré sur le tag, pas juste le premier de la liste ------------------

def test_le_run_surveille_est_celui_du_bon_tag(tmp_path):
    d = _depot_jetable(tmp_path)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "v"], check=True)
    faux_bin = tmp_path / "bin"; faux_bin.mkdir()
    # deux runs : un pour v0.7.3 (plus récent dans la liste), un pour v0.7.4 (le bon) —
    # sans filtre sur headBranch, `head -1` prendrait le run 111 (v0.7.3), jamais le run
    # attendu 222 (v0.7.4). Le faux `gh run watch` refuse tout id ≠ 222.
    env = _faux_gh(faux_bin, f'''
case "$1 $2" in
  "release create") echo "https://github.com/x/y/releases/tag/v{V}"; exit 0 ;;
esac
if [ "$1" = "run" ] && [ "$2" = "list" ]; then
  # fidèle au vrai `gh` : applique le -q (jq) reçu, ne devine rien côté script testé.
  Q=""; a=("$@")
  for ((i=0; i<${{#a[@]}}; i++)); do [ "${{a[$i]}}" = "-q" ] && Q="${{a[$((i+1))]}}"; done
  printf '[{{"databaseId":111,"event":"release","headBranch":"v0.7.3"}},{{"databaseId":222,"event":"release","headBranch":"v{V}"}}]' \\
    | jq -r "$Q"
  exit 0
fi
if [ "$1" = "run" ] && [ "$2" = "watch" ]; then
  [ "$3" = "222" ] && exit 0 || {{ echo "REFUS TEMOIN : run $3 surveille, attendu 222" >&2; exit 1; }}
fi
''')
    faux_home = tmp_path / "home"
    coffre = faux_home / ".config" / "acvram"; coffre.mkdir(parents=True)
    (coffre / "jetons-acvram.sh").write_text("#!/usr/bin/env bash\necho faux-jeton\n", encoding="utf-8")
    (coffre / "jetons-acvram.sh").chmod(0o755)
    env["HOME"] = str(faux_home)
    r = _lancer(d, env, f"v{V}", "--depuis", "5")
    # L'étape 7 (téléchargement réel de la release) n'est pas simulable sans réseau ; seul
    # le choix du run (5-6, l'objet de cette pièce) est sous témoin ici.
    assert "run 222" in r.stdout, r.stdout + r.stderr
    assert "REFUS TEMOIN" not in r.stdout + r.stderr, r.stdout + r.stderr


# ---- (b) --depuis refuse un tag absent ou pas ancêtre de main ------------------------------------

def test_depuis_refuse_si_le_tag_n_existe_pas(tmp_path):
    d = _depot_jetable(tmp_path)
    r = _lancer(d, {}, f"v{V}", "--depuis", "5", "--simule")
    assert r.returncode == 72 and "n'existe pas" in r.stderr, r.stderr


def test_depuis_refuse_si_le_tag_n_est_pas_ancetre_de_main(tmp_path):
    d = _depot_jetable(tmp_path)
    # un tag posé sur une branche à côté, jamais fusionnée dans main
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "-b", "ailleurs"], check=True)
    (d / "autre.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-m", "ailleurs"], check=True)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "v"], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "main"], check=True)
    r = _lancer(d, {}, f"v{V}", "--depuis", "5", "--simule")
    assert r.returncode == 72 and "pas un ancêtre de main" in r.stderr, r.stderr


def test_depuis_accepte_un_tag_valide_et_saute_les_3_premieres_etapes(tmp_path):
    d = _depot_jetable(tmp_path)
    subprocess.run(["git", "-C", str(d), "tag", "-a", f"v{V}", "-m", "v"], check=True)
    r = _lancer(d, {}, f"v{V}", "--depuis", "5", "--simule")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "== 1." not in r.stdout and "== 3." not in r.stdout
    assert "== 5." in r.stdout and "== 6." in r.stdout and "== 7." in r.stdout


def test_depuis_hors_bornes_refuse(tmp_path):
    d = _depot_jetable(tmp_path)
    r = _lancer(d, {}, f"v{V}", "--depuis", "3", "--simule")
    assert r.returncode == 64, r.stderr
