# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""Concentration du routage MoE, à partir de l'histogramme par expert.

L'histogramme lui-même est accumulé sur device par
`MoEBlock._compter_routage` (`engine/model.py`) à chaque pas, jamais lu là.
Ce module ne fait que la mesure À LA DEMANDE, sur ce que
`ACVRamModel.usage_routage()` a rendu — une fonction pure sur des entiers,
qu'ils viennent d'un tenseur CPU ou d'un `.cpu()` fait par l'appelant.

Pourquoi ce chiffre
--------------------

`revue/colibri-hierarchie-experts-disque.md` réclame un compte que nous
n'avions pas : combien d'experts distincts couvrent la majorité du routage
réel. C'est la mesure qui dirait si un cache par expert (comme colibrì)
vaut le détour chez nous — notre placement actuel est par COUCHE entière
(`LayerPlacement.mlp_storage`), pas par expert, donc aucun « taux de succès »
ne peut encore se calculer contre un ensemble résident : la concentration est
la première brique, avant tout cache.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch

__all__ = ["concentration_top_fraction", "sauvegarder", "charger",
           "choisir_residents", "decider_residents"]

# Nom du profil persistant dans le dossier d'un modèle converti — le nom
# porte le geste (bead pds), pas le modèle : un seul fichier par dossier.
NOM_PROFIL = ".acvram_usage.json"

# AUTOPIN de colibrì (colibri.c:11214-11262, cité dans
# revue/colibri-lecture-code.md) exige au moins 5000 sélections d'historique
# avant de décider — sous ce seuil, une couche à peine visitée déciderait sur
# du bruit. Même règle ici, même chiffre : rien ne nous dit qu'un autre serait
# mieux, et en inventer un ferait passer une supposition pour une mesure.
SEUIL_CONFIANCE = 5000


def concentration_top_fraction(compte: torch.Tensor, fraction: float = 0.10) -> float:
    """Part des sélections captée par les ``fraction`` experts les plus chauds.

    ``compte`` : un histogramme par expert (un entier non négatif par expert,
    ordre quelconque — ni trié, ni indexé par identifiant).

    Rend 0.0 si ``compte`` est vide ou si son total est nul : aucune sélection
    n'est un résultat (règle 10), pas une division par zéro déguisée en 1.0
    (« 100 % de rien ») ou en NaN.
    """
    if compte.numel() == 0:
        return 0.0
    total = compte.sum()
    if total.item() == 0:
        return 0.0
    k = max(1, int(round(fraction * compte.numel())))
    plus_chauds = torch.topk(compte, min(k, compte.numel())).values.sum()
    return float((plus_chauds / total).item())


def sauvegarder(usage: dict, chemin: str) -> None:
    """Écrit l'histogramme `{couche: tenseur}` de `ACVRamModel.usage_routage()`
    dans un profil persistant, creux (couches/experts jamais vus omis).

    Format JSON `{"<couche>": {"<expert>": compte, ...}, ...}` — pas le texte
    creux de colibrì (`.coli_usage`) : rien chez nous ne lit ce format à sa
    place, et JSON évite d'inventer un parseur pour un gain de taille qui ne
    sert personne ici. Ce que colibrì en tire — un fichier lisible, qui ne
    grossit que par ce qui a vraiment été sélectionné — est conservé : les
    zéros ne s'écrivent pas."""
    dehors = {}
    for couche, compte in usage.items():
        compte_cpu = compte.detach().to("cpu")
        ligne = {str(e): int(c) for e, c in enumerate(compte_cpu.tolist()) if c}
        if ligne:
            dehors[str(couche)] = ligne
    tmp = f"{chemin}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dehors, f)
    os.replace(tmp, chemin)          # atomique : jamais un profil à moitié écrit


def charger(chemin: str) -> Optional[dict]:
    """Relit un profil écrit par `sauvegarder`. Rend `None`, pas un profil
    vide, si le fichier n'existe pas ou est illisible — un profil absent et un
    profil qui dit « rien n'a jamais été sélectionné » ne sont pas le même
    fait (règle 10 : un échec est un résultat, pas un zéro qui l'imite)."""
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            brut = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return {int(couche): {int(e): c for e, c in ligne.items()}
           for couche, ligne in brut.items()}


def choisir_residents(compte_couche: dict, capacite: int,
                      seuil_confiance: int = SEUIL_CONFIANCE) -> Optional[list]:
    """AUTOPIN d'une couche : les `capacite` experts les plus demandés de
    `compte_couche` (`{expert: compte}`, tel que rendu par `charger`).

    Rend `None` — pas la liste vide, pas un tirage arbitraire — si
    `sum(compte_couche.values()) < seuil_confiance` : sous ce seuil, décider
    reviendrait à épingler sur du bruit (même geste que `LayerPlacement.
    taux_succes = None` : l'absence de mesure ne se comble pas d'une valeur
    qui a l'air d'une mesure). L'appelant garde alors le grain « couche
    entière » (`cached_expert_fraction`) plutôt que ce grain plus fin.
    """
    if sum(compte_couche.values()) < seuil_confiance:
        return None
    classes = sorted(compte_couche.items(), key=lambda kv: (-kv[1], kv[0]))
    return [e for e, _ in classes[:capacite]]


def decider_residents(compte_couche: Optional[dict], n_experts: int,
                      capacite: int,
                      seuil_confiance: int = SEUIL_CONFIANCE
                      ) -> tuple[set, str]:
    """Décision AU CHARGEMENT, pour UNE couche : quels experts résident en
    VRAM. Rend `(residents, source)` — `source` distingue explicitement une
    décision MESURÉE (``"autopin"``, depuis `.acvram_usage`) d'un DÉFAUT
    (``"defaut"``, les `capacite` premiers indices) : un journal qui dirait
    seulement les identifiants ne saurait pas si le plan a appris quelque
    chose ou tiré au sort par ordre d'apparition.

    `compte_couche` : `{expert: compte}` pour CETTE couche (déjà extrait d'un
    profil chargé par `charger()`), ou `None`/`{}` si aucun profil n'existe
    encore — cas normal au premier chargement d'un modèle, pas une erreur.
    """
    if compte_couche:
        residents = choisir_residents(compte_couche, capacite, seuil_confiance)
        if residents is not None:
            return set(residents), "autopin"
    return set(range(min(capacite, n_experts))), "defaut"
