"""
Centralisation des emplacements du parc acvram.

Unifie l'accès aux modèles convertis depuis plusieurs racines
pour éviter les duplications et les divergences.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


# Racines du parc, dans l'ordre de priorite. La premiere suit le montage
# reel du SSD (12/09/2026 : passage Mint→Ubuntu, changement de point de montage).
PARC_ROOTS = [
    _RACINE,
    _RACINE,
]

# Surcharge par variable d'env, deux formes acceptees :
#   ACVRAM_PARC_ROOTS  : plusieurs racines separees par ':'
#   ACVRAM_MODELES     : une seule racine (partage avec racine_modeles.py)
def _get_configured_roots() -> List[str]:
    """Retourne les racines du parc, avec surcharge possible."""
    env = os.environ.get('ACVRAM_PARC_ROOTS')
    if env:
        return env.split(':')
    seul = os.environ.get('ACVRAM_MODELES')
    if seul:
        return [seul] + PARC_ROOTS[1:]
    return PARC_ROOTS

def _get_single_root() -> str:
    """Retourne la racine de référence (première seulement)."""
    return _get_configured_roots()[0]

def get_parc_models(
    single_root_only: bool = False,
    include_location: bool = False
) -> Dict[str, str] | Dict[str, Tuple[str, str]]:
    """
    Retourne tous les modèles convertis avec manifeste.

    Args:
        single_root_only: Si True, exclut la deuxième racine (pour comparabilité)
        include_location: Si True, retourne (chemin, racine) au lieu de juste (chemin)

    Returns:
        Dict[nom_modele, chemin] ou Dict[nom_modele, (chemin, racine)]

    Raises:
        ValueError: Si un modèle existe sous le même nom dans deux racines
    """
    roots = [_get_single_root()] if single_root_only else _get_configured_roots()
    models = {}
    duplicates = []

    for root in roots:
        if not os.path.isdir(root):
            continue

        for entry in os.listdir(root):
            model_path = os.path.join(root, entry)
            if not os.path.isdir(model_path):
                continue

            manifest_path = os.path.join(model_path, 'acvram_manifest.json')
            if not os.path.isfile(manifest_path):
                continue

            # Vérifier la validité du manifeste
            try:
                with open(manifest_path) as f:
                    json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Vérifier les doublons
            if entry in models:
                duplicates.append((entry, models[entry], model_path))

            if include_location:
                models[entry] = (model_path, root)
            else:
                models[entry] = model_path

    # Signaler les doublons
    if duplicates:
        dup_info = '\n'.join(
            f"  {name}: {path1}\n           {path2}"
            for name, path1, path2 in duplicates
        )
        raise ValueError(f"Modèles en doublon dans deux racines:\n{dup_info}")

    return models

def get_all_model_paths(single_root_only: bool = False) -> List[str]:
    """Retourne tous les chemins de modèles convertis."""
    models = get_parc_models(single_root_only=single_root_only)
    return list(models.values())

def get_model_location(model_name: str) -> Optional[Tuple[str, str]]:
    """
    Retourne (chemin, racine) d'un modèle, ou None s'il n'existe pas.
    """
    try:
        models = get_parc_models(include_location=True)
        return models.get(model_name)
    except ValueError:
        # Gestion du doublon : chercher dans la première racine en priorité
        root = _get_single_root()
        path = os.path.join(root, model_name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'acvram_manifest.json')):
            return (path, root)
    return None

# Compatibilité : alias pour les anciens codes
def get_parc_root() -> str:
    """Retourne la racine de référence (première)."""
    return _get_single_root()

def get_all_parc_roots() -> List[str]:
    """Retourne toutes les racines."""
    return _get_configured_roots()
