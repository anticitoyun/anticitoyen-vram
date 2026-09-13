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

import torch

__all__ = ["concentration_top_fraction"]


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
