"""`import acvram` doit résoudre sous CE worktree, pas le dépôt de base.

Le finder d'installation editable est PARTAGÉ entre tous les worktrees
(un seul `.venv`, `anticitoyen-vram/.venv`) et pointe par défaut sur le
dépôt de base (`anticitoyen-vram/acvram`) — un script lancé sans insérer
la racine du worktree en tête de `sys.path` (voir `outils/_racine.py`)
importe silencieusement le mauvais code, sans erreur pour le signaler.
Tombé trois fois le 14/09/2026 avant d'être nommé ici.

Ce test ne peut vérifier QUE le process pytest lui-même (lancé depuis ce
worktree, donc correct par construction du harnais) — sa valeur est de
DOCUMENTER l'invariant attendu et de casser si le mécanisme
d'installation change de façon à le violer par défaut. La vraie garde
contre le piège est `outils/_racine.py`, à importer dans chaque script.
"""
import pathlib

import acvram

RACINE = pathlib.Path(__file__).resolve().parent.parent


def test_acvram_resout_sous_ce_worktree():
    chemin = pathlib.Path(acvram.__file__).resolve()
    assert chemin.is_relative_to(RACINE), (
        f"acvram.__file__ = {chemin} n'est PAS sous ce worktree ({RACINE}) — "
        f"le finder d'installation editable partagé a resolu vers un autre "
        f"depot. Verifier que rien n'a change dans la resolution du paquet "
        f"pour ce process de test.")
