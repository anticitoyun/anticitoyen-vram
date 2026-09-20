"""M-RoPE (Qwen3-VL) : positions à trois axes (t, h, w) et sections entrelacées.

Référence reproduite au bit : transformers ``modeling_qwen3_vl.py`` —
``Qwen3VLModel.get_rope_index`` / ``get_vision_position_ids`` pour les positions,
``Qwen3VLTextRotaryEmbedding.recomposition_frequencies`` pour l'entrelacement
des sections (contrat revue/poste7-go-qwen3vl-parallele-20-09 § 2).

Conventions :
- le texte avance sur les trois axes à la fois (``arange + courant``) ;
- une image de grille (t, h, w) — après fusion spatiale ``merge`` : (t, h//merge,
  w//merge) — pose ``t·h'·w'`` jetons : axe t = courant + indice temporel, axe h =
  courant + ligne, axe w = courant + colonne ; le curseur avance ensuite de
  ``max(h', w')`` seulement, PAS du nombre de jetons ;
- ``delta`` = ``max(positions) + 1 − T`` (le ``rope_deltas`` de transformers, ≤ 0) :
  au décodage, la position M-RoPE d'un jeton vaut sa position 1-D + delta sur les
  trois axes — un seul entier par séquence, le rejeu des graphes n'en sait rien.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

__all__ = ["axes_interleaved", "positions_mrope", "grille_de", "section_depuis"]


def section_depuis(rope_scaling: Optional[dict]) -> Optional[list[int]]:
    """``mrope_section`` d'un ``rope_scaling`` (ou ``rope_parameters``) Qwen3-VL,
    None pour tout modèle sans M-RoPE. Seule la variante ENTRELACÉE est servie :
    une section non entrelacée (Qwen2-VL, blocs contigus) est refusée nommément
    plutôt que tournée de travers."""
    if not rope_scaling:
        return None
    section = rope_scaling.get("mrope_section")
    if not section:
        return None
    if not rope_scaling.get("mrope_interleaved", False):
        raise ValueError(f"mrope_section {list(section)} sans mrope_interleaved : "
                         "seule la variante entrelacée (Qwen3-VL) est servie")
    return [int(s) for s in section]


def axes_interleaved(section: Sequence[int], n_freq: int) -> torch.Tensor:
    """Axe (0 = t, 1 = h, 2 = w) de chaque indice de fréquence, [n_freq] long.

    transformers (``recomposition_frequencies``) part de l'axe t partout, puis
    écrit h sur ``slice(1, 3·section[1], 3)`` et w sur ``slice(2, 3·section[2], 3)`` :
    [24, 20, 20] sur 64 fréquences donne t,h,w,t,h,w,… jusqu'à 60, puis t sur les
    quatre dernières."""
    if len(section) != 3:
        raise ValueError(f"mrope_section attend 3 sections (t, h, w), reçu {list(section)}")
    if sum(section) != n_freq:
        raise ValueError(f"mrope_section {list(section)} somme {sum(section)} ≠ {n_freq} "
                         "fréquences (head_dim / 2)")
    axes = torch.zeros(n_freq, dtype=torch.long)
    for axe, decalage in ((1, 1), (2, 2)):
        axes[decalage:int(section[axe]) * 3:3] = axe
    return axes


def _thw(grid_thw: Any) -> tuple[int, int, int]:
    g = grid_thw
    if isinstance(g, torch.Tensor):
        g = g.reshape(-1).tolist()
    g = [int(v) for v in g]
    if len(g) != 3:
        raise ValueError(f"image_grid_thw attend (t, h, w), reçu {g}")
    return g[0], g[1], g[2]


def grille_de(fragment: Any) -> Any:
    """La grille (t, h, w) d'un fragment d'image : ``supplement["image_grid_thw"]``
    ([1, 3] par image, découpé par server/chat._par_image)."""
    supp = getattr(fragment, "supplement", None) or {}
    g = supp.get("image_grid_thw")
    if g is None:
        raise ValueError("fragment d'image sans supplement['image_grid_thw'] : "
                         "positions M-RoPE incalculables")
    return g


def positions_mrope(prompt_len: int, images: Sequence[tuple], merge: int
                    ) -> tuple[torch.Tensor, int]:
    """Positions [3, prompt_len] (long) et ``delta`` d'une invite.

    ``images`` : (debut, fin, grid_thw) en positions absolues de l'invite, dans
    l'ordre ; ``fin − debut`` doit valoir ``t · (h // merge) · (w // merge)``.
    Sans image : les trois axes valent ``arange(prompt_len)`` et delta = 0."""
    if merge <= 0:
        raise ValueError(f"spatial_merge_size {merge} invalide")
    ims = sorted(((int(d), int(f), g) for d, f, g in images), key=lambda x: x[0])
    morceaux: list[torch.Tensor] = []
    courant = 0            # curseur de position (transformers : current_pos)
    fait = 0               # jetons déjà couverts
    for debut, fin, g in ims:
        if debut < fait or fin > prompt_len or fin <= debut:
            raise ValueError(f"plage image [{debut}, {fin}) hors de l'invite ({prompt_len}) "
                             "ou chevauchante")
        t, h, w = _thw(g)
        gh, gw = h // merge, w // merge
        if t * gh * gw != fin - debut:
            raise ValueError(f"grille ({t}, {h}, {w}) / merge {merge} = {t * gh * gw} jetons, "
                             f"la plage [{debut}, {fin}) en porte {fin - debut}")
        if debut > fait:                                     # texte avant l'image
            n = debut - fait
            morceaux.append((torch.arange(n, dtype=torch.long) + courant).view(1, -1).expand(3, -1))
            courant += n
        tt, hh, ww = torch.meshgrid(torch.arange(t, dtype=torch.long),
                                    torch.arange(gh, dtype=torch.long),
                                    torch.arange(gw, dtype=torch.long), indexing="ij")
        morceaux.append(torch.stack((tt, hh, ww), dim=0).reshape(3, -1) + courant)
        courant += max(gh, gw)
        fait = fin
    if fait < prompt_len:
        n = prompt_len - fait
        morceaux.append((torch.arange(n, dtype=torch.long) + courant).view(1, -1).expand(3, -1))
    if not morceaux:
        return torch.zeros(3, 0, dtype=torch.long), 0
    pos = torch.cat(morceaux, dim=1).contiguous()
    delta = int(pos.max().item()) + 1 - prompt_len if prompt_len else 0
    return pos, delta
