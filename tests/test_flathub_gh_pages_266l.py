"""Pièce 266 l (décision chef 26/09) : le dépôt Flatpak de gh-pages est UN dépôt qui s'accumule, pas un dépôt réécrit
par le dernier job venu. Le 26/09 les jobs flatpak des v0.7.1 et v0.7.2 couraient en même temps : chacun faisait
`rm -rf pages/flatpak && cp -a repo pages/flatpak` d'un dépôt fraîchement exporté, le dernier à pousser gagnait le ref
master quelle que soit sa version, et deux push pouvaient se refuser. Trois gardes : (1) concurrency au job flatpak
(tous runs confondus, sans annulation) ; (2) `repo` amorcé depuis pages/flatpak avant flatpak-builder, prune-depth 1 ;
(3) le ref master ne recule jamais — ref-ne-recule-pas.sh compare la version du tag à celle servie (VERSION)."""
import pathlib
import subprocess

import yaml

RACINE = pathlib.Path(__file__).resolve().parents[1]
RELEASE = RACINE / ".github" / "workflows" / "release.yml"
SCRIPT = RACINE / "packaging" / "flathub" / "ref-ne-recule-pas.sh"


def _flatpak():
    return yaml.safe_load(RELEASE.read_text(encoding="utf-8"))["jobs"]["flatpak"]


def _run():
    return "\n".join(s.get("run", "") for s in _flatpak()["steps"])


def test_266l_un_seul_job_flatpak_a_la_fois_sans_annulation():
    assert _flatpak()["concurrency"] == {"group": "flatpak-gh-pages", "cancel-in-progress": False}


def test_266l_le_depot_servi_amorce_repo_avant_la_construction():
    run = _run()
    assert run.index("git worktree add pages gh-pages") < run.index("flatpak-builder --user"), "gh-pages se lit AVANT de construire"
    ligne = next(l for l in run.splitlines() if "cp -a pages/flatpak repo" in l)      # 266 m : dans un `if`, suivi du mkdir
    assert "[ -d pages/flatpak/objects ]" in ligne and ligne.index("[ -d pages/flatpak/objects ]") < ligne.index("cp -a pages/flatpak repo")
    assert run.count("--prune --prune-depth=1") == 2, "les deux branches (signée ou non) élaguent"


def test_266l_le_ref_master_ne_recule_pas_dans_le_job():
    run = _run()
    assert 'packaging/flathub/ref-ne-recule-pas.sh "$V" pages/flatpak/VERSION' in run
    assert 'echo "$V" > repo/VERSION' in run, "la version servie est écrite avec le dépôt"
    for geste in ("flatpak-builder --user", "flatpak build-sign", 'git push --force origin "$NOUVEAU:refs/heads/gh-pages"'):
        i = run.index(geste)
        assert 'if [ "$CONSTRUIRE" = 1 ]; then' in run[max(0, i - 1300):i], geste + " hors de la garde CONSTRUIRE"
    assert "::warning::gh-pages sert une version plus récente" in run
    assert 'gh release upload "$TAG" "acvram-$V.flatpakref"' in run.split('refs/heads/gh-pages"')[1], \
        "le .flatpakref est joint même quand le dépôt n'est pas touché"


def _script(v, contenu, tmp):
    f = tmp / "VERSION"
    if contenu is not None:
        f.write_text(contenu, encoding="utf-8")
    r = subprocess.run(["bash", str(SCRIPT), v, str(f)], capture_output=True, text=True, timeout=20)
    return r.returncode, r.stdout.strip()


def test_266l_ref_ne_recule_pas_sait_dire_construire_et_sauter(tmp_path):
    assert _script("0.7.2", None, tmp_path)[0] == 0                    # dépôt vide
    assert _script("0.7.2", "0.7.1\n", tmp_path)[0] == 0                # plus récente
    assert _script("0.7.2", "0.7.2\n", tmp_path)[0] == 0                # même version rejouée
    assert _script("0.7.10", "0.7.9\n", tmp_path)[0] == 0               # tri de versions, pas lexical
    code, motif = _script("0.7.1", "0.7.2\n", tmp_path)                 # release ancienne relancée
    assert code == 3 and "ne recule pas" in motif and "0.7.2" in motif, (code, motif)
    assert _script("0.7.9", "0.7.10\n", tmp_path)[0] == 3
    assert _script("0.7.2", "n'importe quoi\n", tmp_path)[0] == 0       # VERSION illisible : construire, dit


def test_266l_gh_pages_est_un_commit_orphelin_pousse_de_force():
    """L'historique git de gh-pages ne s'accumule pas (≈ 100 Mio d'objets par version) : un commit-tree SANS parent à
    chaque publication, poussé --force sur refs/heads/gh-pages seulement ; plus de `git -C pages push origin gh-pages`."""
    run = _run()
    ligne = next(l for l in run.splitlines() if "NOUVEAU=$(git -C pages" in l)
    assert "commit-tree" in ligne and " -p " not in ligne and "github-actions[bot]" in ligne
    assert 'git push --force origin "$NOUVEAU:refs/heads/gh-pages"' in run
    assert "git -C pages push origin gh-pages" not in run and "git -C pages commit" not in run


def test_266m_le_depot_copie_depuis_gh_pages_retrouve_ses_dossiers_vides():
    """Run 36243027577 (premier run 266 l) : `cp -a pages/flatpak repo` depuis un clone git → refs/remotes absent (git ne
    garde pas les dossiers vides) → build-update-repo : « Listing refs: opendir(refs/remotes): No such file ». Témoin local
    du 26/09 : copie brute du clone gh-pages v0.7.2 → même erreur, code 1 ; copie + mkdir -p des quatre dossiers → « Updating
    summary », code 0 ; avec des .keep dedans → code 0 aussi. Le job recrée les dossiers à l'amorçage et pose des .keep à
    la publication pour que gh-pages les conserve."""
    run = _run()
    i = run.index("cp -a pages/flatpak repo")
    assert "mkdir -p repo/refs/remotes repo/refs/mirrors repo/tmp repo/state" in run[i:i + 200], "mkdir juste après la copie"
    j = run.index("cp -a repo pages/flatpak")
    assert 'for d in refs/remotes refs/mirrors tmp state; do mkdir -p "pages/flatpak/$d" && touch "pages/flatpak/$d/.keep"; done' in run[j:j + 400]
    assert run.index("mkdir -p repo/refs/remotes") < run.index("flatpak-builder --user"), "avant la construction"
