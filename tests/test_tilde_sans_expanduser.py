"""Une chaîne « ~/… » n'est jamais développée par Python lui-même.

La purge (25/09) a remplacé `/home/<utilisateur>/…` par `~/…` dans les
scripts de scratchpad. En bash nu ou dans le shell interactif, `~` se
développe ; passée telle quelle à `sys.path.insert` ou `open` en Python,
c'est un composant de chemin littéral `~` — `ModuleNotFoundError` ou
`FileNotFoundError` immédiats (pièce 192 bis, `banc-chat-openai.py`).

Ce test lit l'AST de chaque fichier `.py` suivi et refuse toute chaîne
littérale commençant par `~/` passée en argument positionnel à
`sys.path.insert` (2e argument) ou à `open` (1er argument), sauf si elle est
elle-même le résultat direct d'un appel à `os.path.expanduser` /
`pathlib.Path.expanduser` — auquel cas ce n'est plus une chaîne littérale
nue à cet endroit, donc rien à détecter (l'appel expanduser est un autre
nœud AST, pas un argument direct de ce type).
"""
import ast
import pathlib
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent


def fichiers_suivis():
    sortie = subprocess.run(
        ["git", "-C", str(RACINE), "ls-files", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [RACINE / l for l in sortie.splitlines() if l]


def chaine_tilde_nue(noeud):
    return (
        isinstance(noeud, ast.Constant)
        and isinstance(noeud.value, str)
        and noeud.value.startswith("~/")
    )


def cible_du_call(noeud):
    f = noeud.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


class Visiteur(ast.NodeVisitor):
    def __init__(self):
        self.trouves = []

    def visit_Call(self, noeud):
        nom = cible_du_call(noeud)
        if nom == "insert" and len(noeud.args) >= 2 and chaine_tilde_nue(noeud.args[1]):
            self.trouves.append(noeud.lineno)
        elif nom == "open" and noeud.args and chaine_tilde_nue(noeud.args[0]):
            self.trouves.append(noeud.lineno)
        self.generic_visit(noeud)


@pytest.mark.parametrize("chemin", fichiers_suivis(), ids=lambda p: str(p.relative_to(RACINE)))
def test_pas_de_tilde_nu_dans_insert_ou_open(chemin):
    try:
        arbre = ast.parse(chemin.read_text(encoding="utf-8"), filename=str(chemin))
    except SyntaxError:
        pytest.skip("fichier suivi non syntaxiquement valide (hors portée de ce test)")
    v = Visiteur()
    v.visit(arbre)
    assert not v.trouves, (
        f"{chemin.relative_to(RACINE)} : chaine '~/...' nue (non "
        f"os.path.expanduser) passee a sys.path.insert ou open, ligne(s) "
        f"{v.trouves} — ~ ne se developpe pas en Python"
    )
