"""Pièce 252 (chef, 26/09) : `outils/session-acvram-worktree.sh` doit
désigner le worktree de la branche `origin/<prénom>[-*]` la plus récemment
COMMITÉE, pas la première ligne d'un `git worktree list | grep -i $p`
(poste2 et poste4 ont repris sur un vieux worktree au redémarrage à cause
de ce piège — l'ordre de `git worktree list` est l'ordre d'AJOUT, jamais
celui de fraîcheur)."""
import os
import pathlib
import subprocess
import time

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "outils/session-acvram-worktree.sh"


def _commit(depot, message, quand_epoch):
    date = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(quand_epoch)) + " +0000"
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    (depot / "f.txt").write_text(message)
    subprocess.run(["git", "add", "f.txt"], cwd=depot, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=depot, env=env, check=True)


@pytest.fixture()
def depot_git_init_main(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    _commit(clone, "premier", int(time.time()) - 4 * 86400)
    subprocess.run(["git", "branch", "-m", "main"], cwd=clone, check=True)
    subprocess.run(["git", "push", "-q", "-u", "origin", "main"], cwd=clone, check=True)
    return clone


def test_designe_le_worktree_le_plus_recent_qui_en_a_un(tmp_path, depot_git_init_main):
    clone = depot_git_init_main
    maintenant = int(time.time())
    attendu = None
    for nom, age, avec_worktree in [
        ("poste3-ancienne", maintenant - 3 * 86400, True),
        ("poste3-moyenne", maintenant - 1 * 86400, True),
        ("poste3-recente-sans-worktree", maintenant - 60, False),
    ]:
        subprocess.run(["git", "checkout", "-q", "-b", nom, "main"], cwd=clone, check=True)
        _commit(clone, f"commit {nom}", age)
        subprocess.run(["git", "push", "-q", "origin", nom], cwd=clone, check=True)
        # revenir sur main : sinon `worktree add` refuse une branche déjà
        # cochée dans CE worktree (le clone lui-même compte comme worktree).
        subprocess.run(["git", "checkout", "-q", "main"], cwd=clone, check=True)
        if avec_worktree:
            wt = tmp_path / f"travail-{nom}"
            subprocess.run(["git", "worktree", "add", "-q", str(wt), nom], cwd=clone, check=True)
            if nom == "poste3-moyenne":
                attendu = str(wt)

    r = subprocess.run(["bash", str(SCRIPT), "poste3", str(clone)],
                        capture_output=True, text=True, check=True)
    assert r.stdout.strip() == attendu, (
        f"attendu le worktree de poste3-moyenne (la plus recente AVEC worktree), "
        f"recu {r.stdout.strip()!r} — poste3-recente-sans-worktree n'a pas de "
        f"worktree local et doit etre saute meme si elle est la plus recente.")


def test_aucune_branche_du_prenom_rend_code_un(depot_git_init_main):
    r = subprocess.run(["bash", str(SCRIPT), "poste2", str(depot_git_init_main)],
                        capture_output=True, text=True)
    assert r.returncode == 1
    assert r.stdout.strip() == ""


def test_ne_confond_pas_deux_prenoms_qui_se_recouvrent(tmp_path, depot_git_init_main):
    """poste3 et poste4 : le motif doit s'arrêter au tiret, jamais matcher
    "poste4-xxx" pour le prénom "poste3" (piège du `grep -i` d'avant)."""
    clone = depot_git_init_main
    maintenant = int(time.time())
    subprocess.run(["git", "checkout", "-q", "-b", "poste4-244", "main"], cwd=clone, check=True)
    _commit(clone, "poste4", maintenant - 10)
    subprocess.run(["git", "push", "-q", "origin", "poste4-244"], cwd=clone, check=True)
    subprocess.run(["git", "checkout", "-q", "main"], cwd=clone, check=True)
    subprocess.run(["git", "worktree", "add", "-q", str(tmp_path / "wt-poste4"),
                     "poste4-244"], cwd=clone, check=True)

    r = subprocess.run(["bash", str(SCRIPT), "poste3", str(clone)],
                        capture_output=True, text=True)
    assert r.returncode == 1, "poste4-244 n'est pas une branche de poste3, doit rester introuvable"
