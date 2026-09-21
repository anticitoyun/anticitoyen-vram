"""Tête de prédiction multi-jetons (MTP), utilisée comme brouillon spéculatif.

Les modèles Qwen3.5/3.6/3.8 et DeepSeek livrent, à la fin de leur pile, une ou
plusieurs couches ``nextn`` : un bloc de transformeur complet précédé d'une
projection qui mélange l'état caché de la position courante et le plongement du
jeton qui vient d'être émis. Entraînée à prédire le jeton *suivant le suivant*,
elle sert de brouillon presque gratuit — un soixante-quatrième du modèle sur un
27B, là où un modèle brouillon séparé coûtait un modèle entier et faisait
tomber le débit de 152 à 30 t/s.

Le schéma est celui de DeepSeek-V3 :

    h' = eh_proj( [ enorm(embed(t)) ; hnorm(h) ] )
    h" = bloc(h')
    logits = lm_head( shared_head_norm(h") )

où ``h`` est l'état caché de la dernière couche du modèle principal à la
position précédente. Pour un brouillon de plusieurs jetons, la tête se relit
elle-même : le ``h"`` qu'elle produit devient le ``h`` du jeton suivant.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..memory.kvcache import PagedKVCache
from .layers import RMSNorm
from .model import ForwardBatch


class MTPHead(nn.Module):
    """Une couche ``nextn`` prête à engendrer des jetons brouillons."""

    def __init__(self, layer: nn.Module, enorm: RMSNorm, hnorm: RMSNorm,
                 eh_proj: nn.Module, final_norm: RMSNorm,
                 cache: Optional[PagedKVCache], device: torch.device) -> None:
        super().__init__()
        self.layer = layer
        self.enorm = enorm
        self.hnorm = hnorm
        self.eh_proj = eh_proj
        self.final_norm = final_norm
        self.cache = cache
        self.device = device

    @torch.inference_mode()
    def forward(self, embeds: torch.Tensor, hidden: torch.Tensor,
                batch: ForwardBatch) -> torch.Tensor:
        """Rend l'état caché normalisé, prêt pour ``lm_head``.

        ``embeds`` et ``hidden`` sont alignés : la ligne *i* porte le plongement
        du jeton émis en position *i* et l'état caché de la position *i-1*.
        """
        e = self.enorm(embeds.to(self.device))
        h = self.hnorm(hidden.to(self.device))
        # Plongement d'abord, état caché ensuite : l'ordre inverse donne zéro
        # jeton accepté, celui-ci en donne la moitié en forçage enseignant.
        x = self.eh_proj(torch.cat([e, h], dim=-1))
        x = self.layer(x, batch, self.cache)
        return self.final_norm(x)


def cles_mtp(manifest: dict) -> list[str]:
    """Indices des têtes MTP présentes dans un manifeste, dans l'ordre."""
    vus = set()
    for nom in manifest.get("tensors", ()):
        if nom.startswith("model.mtp."):
            try:
                vus.add(int(nom.split(".")[2]))
            except (IndexError, ValueError):
                continue
    return sorted(vus)
