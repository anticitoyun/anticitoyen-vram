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


def est_tenseur_mtp(nom: str) -> bool:
    """Tenseur d'une tête MTP, dans les deux conventions de `cles_mtp` (partagés ``mtp.*`` compris). Pièce 201 : le
    compte des poids chargés après la borne du KV (`loader._octets_annexes`) les retire sous ``ACVRAM_MTP=non``."""
    return nom.startswith(("model.mtp.", "mtp."))


def cles_mtp(manifest: dict) -> list[int]:
    """Indices des têtes MTP présentes dans un manifeste, dans l'ordre. Deux conventions :
    ``model.mtp.<n>.`` (GGUF renommé par quant/gguf.py, DeepSeek) et, pièce 105, celle des checkpoints HF de la
    famille Qwen3.5 (``mtp.layers.<n>.`` + ``mtp.fc``/``mtp.norm``/``mtp.pre_fc_norm_*`` partagés) — sans elle, les
    15 tenseurs MTP de Qwen3.8-27B étaient convertis mais jamais chargés, et ``auto`` retombait sur n-gram."""
    vus = set()
    for nom in manifest.get("tensors", ()):
        if nom.startswith("model.mtp.") or nom.startswith("mtp.layers."):
            try:
                vus.add(int(nom.split(".")[2]))
            except (IndexError, ValueError):
                continue
    return sorted(vus)


# Normes « zéro-centrées » (x·(1 + w) dans la référence Qwen3.5) hors couche : la conversion (quant/convert.py,
# _NORMES_ZERO_CENTREES, par suffixe) décale celles de mtp.layers.<n>.* mais PAS ces trois-là. Décalées ici, au
# chargement, sauf si un manifeste futur déclare les avoir décalées (`mtp_normes_decalees`).
NORMES_QWEN35_A_DECALER = ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight")


def noms_mtp(manifest: dict, n: int) -> dict:
    """Noms des tenseurs de la tête ``n`` selon la convention du manifeste : ``bloc`` (préfixe de la couche de
    transformeur), ``eh`` (projection [plongement ; état caché]), ``enorm``, ``hnorm``, ``fin`` (candidats de la
    norme finale, dans l'ordre), ``convention`` (deepseek | qwen35)."""
    p = f"model.mtp.{n}."
    if any(k.startswith(p) for k in manifest.get("tensors", ())):
        return {"convention": "deepseek", "bloc": p, "eh": p + "eh_proj.weight", "enorm": p + "enorm.weight",
                "hnorm": p + "hnorm.weight", "fin": (p + "shared_head_norm.weight", p + "norm.weight")}
    return {"convention": "qwen35", "bloc": f"mtp.layers.{n}.", "eh": "mtp.fc.weight",
            "enorm": "mtp.pre_fc_norm_embedding.weight", "hnorm": "mtp.pre_fc_norm_hidden.weight",
            "fin": ("mtp.norm.weight",)}


def norme_mtp(nom: str, t: torch.Tensor, convention: str, manifest: dict) -> torch.Tensor:
    """Poids d'une norme MTP prêt pour notre RMSNorm (x·w) : +1 sur les trois normes Qwen3.5 hors couche
    (voir NORMES_QWEN35_A_DECALER), inchangé ailleurs."""
    if convention == "qwen35" and nom in NORMES_QWEN35_A_DECALER and not manifest.get("mtp_normes_decalees"):
        return (t.to(torch.float32) + 1.0).to(t.dtype)
    return t
