"""Insère la racine DE CE WORKTREE en tête de `sys.path`.

Sans ceci, `python outils/script.py` met le dossier du SCRIPT (`outils/`)
en `sys.path[0]`, pas la racine du dépôt — un `import acvram` retombe
alors sur l'installation editable PARTAGÉE entre worktrees (le finder du
venv commun pointe sur le dépôt DE BASE, `anticitoyen-vram/acvram`), pas
sur ce worktree. Un correctif fait ici peut alors ne jamais s'exécuter,
sans qu'aucune erreur ne le signale — piège tombé sur trois sessions le
14/09/2026 (Manon, `outils/campagne-a6-snrfloor0.py`, entre autres).

Usage, en tête de tout script sous `outils/` qui importe `acvram` :

    import sys as _s, pathlib as _p
    _s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
    # ou, equivalent et plus court :
    import _racine  # noqa: F401  (l'import seul suffit, l'effet de bord fait le travail)

`_racine.py` doit vivre dans `outils/` : `import _racine` ne fonctionne
que si `outils/` est déjà sur `sys.path` (ce qui est vrai pour
`python outils/script.py`, puisque `sys.path[0]` est le dossier du
script). Il insère alors la racine du DÉPÔT (le parent d'`outils/`) en
position 0, avant toute recherche dans les paquets installés.
"""
import sys as _s
from pathlib import Path as _Path

_RACINE = str(_Path(__file__).resolve().parent.parent)
if _RACINE not in _s.path:
    _s.path.insert(0, _RACINE)
