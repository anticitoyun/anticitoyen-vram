"""Racines partagées, jamais codées en dur dans un script de campagne —
poste7 (revue/poste7-reprise-15-09-b.md §8, 15/09) : la disposition des
disques a changé deux fois le même jour, un chemin en dur y casse en
silence, et chaque chemin codé en dur est un chemin qui ne tourne pour
personne d'autre (tests/test_depot_sans_identite.py, le cliquet).

Usage dans un script de campagne (sibling de outils/, pas un paquet) :

    from _chemins import modeles, sorties
    racine = modeles()          # ACVRAM_MODELES, ou son défaut partagé
    out = sorties() / "mon-resultat.json"   # ACVRAM_SORTIES, ou sous ce worktree
"""
import os
import sys
from pathlib import Path

_ICI = Path(__file__).resolve().parent
if str(_ICI) not in sys.path:      # les deux conventions du dépôt coexistent :
    sys.path.insert(0, str(_ICI))  # sibling nu (`import regime`) et paquet
                                    # (`from outils.racine_modeles import ...`)
                                    # — ce module marche quel que soit celui
                                    # de l'appelant.
import racine_modeles  # noqa: E402

_DEFAUT_SORTIES = _ICI.parent / "acvram-memoire" / "corpus"


def modeles() -> str:
    """Racine du parc de modèles convertis. Un seul littéral pour toute
    valeur par défaut vit dans racine_modeles.py — ce module ne le
    duplique pas, il le relaie."""
    return racine_modeles.MODELES


def sorties() -> Path:
    """Racine où écrire les résultats d'une campagne. Le défaut n'est
    JAMAIS un chemin absolu fixe : `acvram-memoire/corpus` SOUS CE
    WORKTREE, dérivé de `__file__` — chaque worktree a le sien, sans
    littéral à mettre à jour quand un disque change de point de
    montage."""
    return Path(os.environ.get("ACVRAM_SORTIES", str(_DEFAUT_SORTIES)))
