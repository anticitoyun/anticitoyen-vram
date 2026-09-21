"""LFM2 — bloc « short conv » des modèles Liquid (lfm2, lfm2_moe).

Traduction du graphe llama.cpp (src/models/lfm2.cpp) : ``in_proj`` produit
[b | c | x] ; ``bx = b·x`` passe une convolution causale depthwise (noyau
L_cache, sans biais, à état de L_cache−1 colonnes) ; ``y = c · conv(bx)`` ;
``out_proj``. Même interface que les autres couches à état : ``forward(x,
état) -> (y, état)`` et le chemin à formes fixes pour les graphes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LFM2ShortConv"]


class LFM2ShortConv(nn.Module):
    def __init__(self, in_proj: nn.Module, out_proj: nn.Module,
                 conv_weight: torch.Tensor, dim: int) -> None:
        super().__init__()
        self.in_proj, self.out_proj = in_proj, out_proj
        self.conv = nn.Parameter(conv_weight.to(torch.float32), requires_grad=False)
        self.dim = dim
        self.kernel = conv_weight.shape[-1]

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        t = x.shape[0]
        bcx = self.in_proj(x)
        b, c, xx = bcx.split(self.dim, dim=-1)
        bx = (b.to(torch.float32) * xx.to(torch.float32)).t().unsqueeze(0)  # [1, d, t]
        if state is not None:
            seq = torch.cat([state.unsqueeze(0), bx], dim=-1)
        else:
            seq = F.pad(bx, (self.kernel - 1, 0))
        new_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        y = F.conv1d(seq, self.conv.unsqueeze(1), groups=self.dim)[0].t()  # [t, d]
        y = (c.to(torch.float32) * y).to(x.dtype)
        return self.out_proj(y), new_state

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    def new_static(self, device: torch.device) -> dict:
        return {"conv": torch.zeros(self.dim, self.kernel - 1,
                                    dtype=torch.float32, device=device)}

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["conv"].zero_()
        else:
            st["conv"].copy_(etat)

    @staticmethod
    def static_export(st: dict) -> torch.Tensor:
        return st["conv"].clone()

    def decode_static(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        y, new_state = self.forward(x, st["conv"])
        st["conv"].copy_(new_state)
        return y
