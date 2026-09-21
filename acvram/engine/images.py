"""Fragments d'image d'une requête interne (contrat sage-go-multimodal-organisation-20-09 § 2).

L'API prépare, le moteur consomme : une séquence porte ``images``, liste de
``ImageFragment`` dans l'ordre d'apparition dans ``prompt_ids``. Chaque
fragment nomme la plage ``[debut, fin)`` des jetons image DÉJÀ expansés par
l'``AutoProcessor`` (N jetons, comptés dans ``max_model_len`` comme du texte),
les ``pixel_values`` que la tour de vision doit voir, et le sha256 de ces
pixels — que le moteur intègre à la clé du cache de préfixe (jetons +
sha256), pour que deux images différentes sous les mêmes jetons ne partagent
jamais un préfixe.

Ce module ne dépend que de torch : importable par le moteur comme par le
serveur, sans transformers ni Pillow.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["ImageFragment", "sha256_pixel_values"]


def sha256_pixel_values(pixel_values: Any) -> str:
    """sha256 hexadécimal des octets bruts du tenseur (dtype compris)."""
    import torch

    t = pixel_values
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    t = t.detach().cpu().contiguous().flatten()
    if t.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    octets = t.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(octets).hexdigest()


@dataclass
class ImageFragment:
    """Une image de la requête : ``prompt_ids[debut:fin]`` sont ses jetons."""
    debut: int
    fin: int
    pixel_values: Any                       # torch.Tensor, forme du processeur
    sha256: str = ""
    # Tenseurs annexes rendus par le processeur pour CETTE image (Qwen :
    # ``image_grid_thw``), dont la tour de vision a besoin ; vide sinon.
    supplement: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fin <= self.debut:
            raise ValueError(f"fragment d'image vide : [{self.debut}, {self.fin})")
        if not self.sha256:
            self.sha256 = sha256_pixel_values(self.pixel_values)

    @property
    def n_jetons(self) -> int:
        return self.fin - self.debut
