"""Gated DeltaNet — les couches à récurrence linéaire des hybrides Qwen3-Next
(« qwen35 » côté GGUF), dont les modèles « kimi » du parc dérivent.

La règle delta elle-même — le cœur mathématique, facile à se tromper et
invérifiable à l'œil — vient de l'implémentation de référence de
``transformers`` (`torch_chunk_gated_delta_rule` au prefill,
`torch_recurrent_gated_delta_rule` au décodage), comme la reconstruction EXL3
vient d'exllamav3 : dépendance optionnelle, mathématique garantie. Ce module
fournit ce qui l'entoure : les projections (nos ``QuantLinear``), la
convolution causale à état, la normalisation gated et l'état par séquence.

L'état d'une séquence pour une couche : ``(conv_state [conv_dim, k-1],
S [1, num_v_heads, d_k, d_v])`` — quelques mégaoctets, porté par le moteur,
hors du cache paginé (ces couches n'ont pas de KV).
"""

from __future__ import annotations

from typing import Optional

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedDeltaNet", "gdn_available"]


def _refs():
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
    return torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule


def gdn_available() -> bool:
    try:
        _refs()
        return True
    except ImportError:
        return False


class GatedDeltaNet(nn.Module):
    def __init__(self, qkv: nn.Module, gate: nn.Module, alpha: nn.Module,
                 beta: nn.Module, out: nn.Module,
                 conv_weight: torch.Tensor,        # [conv_dim, kernel]
                 dt_bias: torch.Tensor,            # [num_v_heads]
                 a_log: torch.Tensor,              # [num_v_heads]
                 norm_weight: torch.Tensor,        # [head_v_dim]
                 num_k_heads: int, num_v_heads: int,
                 head_k_dim: int, head_v_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.qkv, self.gate = qkv, gate
        self.alpha, self.beta_proj, self.out_proj = alpha, beta, out
        self.conv_weight = nn.Parameter(conv_weight, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias.float(), requires_grad=False)
        self.a_log = nn.Parameter(a_log.float(), requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.nk, self.nv = num_k_heads, num_v_heads
        self.dk, self.dv = head_k_dim, head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = conv_weight.shape[-1]
        self.eps = eps

    # -- normalisation gated (RMSNorm de la sortie, porte SiLU(z)) --------
    def _norm_gated(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        x32 = x32 * self.norm_weight.to(torch.float32)
        return (x32 * F.silu(z.to(torch.float32))).to(x.dtype)

    def forward(self, x: torch.Tensor,
                state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, nouvel état)."""
        chunk_rule, recurrent_rule = _refs()
        t = x.shape[0]
        decode = (t == 1 and state is not None)

        # toute la récurrence se calcule en float32 : la règle delta cumule
        # des produits d'état où le bfloat16 dérive vite
        qkv = self.qkv(x).to(torch.float32)                 # [t, conv_dim]
        z = self.gate(x).to(torch.float32)                  # [t, value_dim]
        b = self.beta_proj(x).to(torch.float32)             # [t, nv]
        a = self.alpha(x).to(torch.float32)                 # [t, nv]

        # convolution causale depthwise, avec état (kernel-1 colonnes)
        seq = qkv.t().unsqueeze(0)                          # [1, conv_dim, t]
        if state is not None:
            conv_state = state[0]
            seq = torch.cat([conv_state.unsqueeze(0), seq], dim=-1)
        else:
            seq = F.pad(seq, (self.kernel - 1, 0))
        new_conv_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        conv = F.conv1d(seq, self.conv_weight.unsqueeze(1),
                        groups=self.conv_dim)               # [1, conv_dim, t]
        conv = F.silu(conv)

        mixed = conv.transpose(1, 2)                        # [1, t, conv_dim]
        q, k, v = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, t, self.nk, self.dk)
        k = k.reshape(1, t, self.nk, self.dk)
        v = v.reshape(1, t, self.nv, self.dv)

        beta = b.sigmoid().unsqueeze(0)                     # [1, t, nv]
        g = (-self.a_log.exp() * F.softplus(a + self.dt_bias)).unsqueeze(0)
        if self.nv // self.nk > 1:
            q = q.repeat_interleave(self.nv // self.nk, dim=2)
            k = k.repeat_interleave(self.nv // self.nk, dim=2)

        s_prev = state[1] if state is not None else None
        rule = recurrent_rule if decode else chunk_rule
        core, s_new = rule(q, k, v, g=g, beta=beta,
                           initial_state=s_prev, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)

        core = core.reshape(-1, self.dv)
        y = self._norm_gated(core, z.reshape(-1, self.dv))
        y = y.reshape(t, self.value_dim)
        return self.out_proj(y.to(x.dtype)), (new_conv_state, s_new)

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    # La règle delta de référence est déjà à formes fixes pour t = 1 : on la
    # rejoue telle quelle et l'on recopie ses sorties dans les tampons fixes
    # — même mathématique, mêmes noyaux, donc mêmes arrondis que ``forward``.
    def new_static(self, device: torch.device) -> dict:
        return {"conv": torch.zeros(self.conv_dim, self.kernel - 1,
                                    dtype=torch.float32, device=device),
                "S": torch.zeros(1, self.nv, self.dk, self.dv,
                                 dtype=torch.float32, device=device)}

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["conv"].zero_(); st["S"].zero_()
            return
        st["conv"].copy_(etat[0]); st["S"].copy_(etat[1])

    @staticmethod
    def static_export(st: dict) -> tuple:
        return (st["conv"].clone(), st["S"].clone())

    def decode_static(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        y, (conv, S) = self.forward(x, (st["conv"], st["S"]))
        st["conv"].copy_(conv)
        st["S"].copy_(S.to(torch.float32))
        return y
