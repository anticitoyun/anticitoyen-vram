"""KDA — Kimi Delta Attention (architecture GGUF « kimi-linear »).

Traduction op par op du graphe de llama.cpp (src/models/kimi-linear.cpp +
delta-net-base.cpp), notre seul exécuteur de référence local pour ce format.
Conventions du convertisseur, déjà appliquées dans le GGUF :
- ``ssm_a`` stocke **−exp(A_log)** par tête (comme qwen35) — utilisé tel quel
  ici, car la formule llama.cpp est ``g1 = ssm_a · softplus(f_b(f_a(x)) + dt)``.
- Convolutions causales séparées pour q, k, v (noyau 4, SiLU), chacune avec
  son état de ``kernel−1`` colonnes.
- q et k L2-normalisés par tête ; q mis à l'échelle 1/√d.
- Récurrence delta par canal : ``S ← S ⊙ exp(g1)`` sur l'**axe clé** (la
  convention fla/vLLM — tranché par bissection contre llama.cpp : l'autre axe
  dérive en boucles après ~7 jetons), puis ``S += β(v−S·k)⊗k ; o = S·q``.
- Sortie : RMSNorm(ssm_norm) par tête × sigmoid(g_b(g_a(x))), puis out_proj.

L'état d'une séquence : ``(conv_q, conv_k, conv_v [d_inner, kernel−1],
S [heads, d, d] float32)`` — porté par le moteur, hors cache paginé.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KimiDeltaAttention"]


class KimiDeltaAttention(nn.Module):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module,
                 out_proj: nn.Module,
                 f_a: nn.Module, f_b: nn.Module,
                 g_a: nn.Module, g_b: nn.Module,
                 beta: nn.Module,
                 conv_q: torch.Tensor, conv_k: torch.Tensor,  # [d_inner, kernel]
                 conv_v: torch.Tensor,
                 dt_bias: torch.Tensor,                       # [d_inner]
                 a: torch.Tensor,                             # [heads] = -exp(A_log)
                 norm_weight: torch.Tensor,                   # [head_dim]
                 num_heads: int, head_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = q_proj, k_proj, v_proj
        self.out_proj = out_proj
        self.f_a, self.f_b, self.g_a, self.g_b = f_a, f_b, g_a, g_b
        self.beta_proj = beta
        self.conv_q = nn.Parameter(conv_q, requires_grad=False)
        self.conv_k = nn.Parameter(conv_k, requires_grad=False)
        self.conv_v = nn.Parameter(conv_v, requires_grad=False)
        self.dt_bias = nn.Parameter(dt_bias.float(), requires_grad=False)
        self.a = nn.Parameter(a.float().reshape(-1), requires_grad=False)
        self.norm_weight = nn.Parameter(norm_weight, requires_grad=False)
        self.nh, self.d = num_heads, head_dim
        self.d_inner = num_heads * head_dim
        self.kernel = conv_q.shape[-1]
        self.eps = eps

    def _conv(self, x: torch.Tensor, w: torch.Tensor,
              state: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Conv1d causale depthwise + SiLU ; rend (sortie [t, d_inner], état)."""
        seq = x.t().unsqueeze(0)                              # [1, d_inner, t]
        if state is not None:
            seq = torch.cat([state.unsqueeze(0), seq], dim=-1)
        else:
            seq = F.pad(seq, (self.kernel - 1, 0))
        new_state = seq[0, :, -(self.kernel - 1):].detach().clone()
        y = F.conv1d(seq, w.unsqueeze(1), groups=self.d_inner)
        return F.silu(y[0].t()), new_state                    # [t, d_inner]

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor,
                state: Optional[tuple] = None
                ) -> tuple[torch.Tensor, tuple]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, nouvel état)."""
        t = x.shape[0]
        cq = ck = cv = None
        s_prev = None
        if state is not None:
            cq, ck, cv, s_prev = state

        q, new_cq = self._conv(self.q_proj(x).to(torch.float32), self.conv_q.float(), cq)
        k, new_ck = self._conv(self.k_proj(x).to(torch.float32), self.conv_k.float(), ck)
        v, new_cv = self._conv(self.v_proj(x).to(torch.float32), self.conv_v.float(), cv)

        q = self._l2norm(q.reshape(t, self.nh, self.d))
        k = self._l2norm(k.reshape(t, self.nh, self.d))
        v = v.reshape(t, self.nh, self.d)
        q = q * (self.d ** -0.5)

        g1 = F.softplus(self.f_b(self.f_a(x)).to(torch.float32) + self.dt_bias)
        g1 = g1.reshape(t, self.nh, self.d) * self.a.reshape(1, self.nh, 1)
        beta = torch.sigmoid(self.beta_proj(x).to(torch.float32))   # [t, nh]

        S = s_prev if s_prev is not None else torch.zeros(
            self.nh, self.d, self.d, dtype=torch.float32, device=x.device)
        sorties = torch.empty(t, self.nh, self.d,
                              dtype=torch.float32, device=x.device)
        for i in range(t):
            S = S * torch.exp(g1[i]).unsqueeze(-2)            # décroissance (axe clé)
            pred = torch.einsum('hij,hj->hi', S, k[i])
            d = beta[i].unsqueeze(-1) * (v[i] - pred)
            S = S + d.unsqueeze(-1) * k[i].unsqueeze(-2)
            sorties[i] = torch.einsum('hij,hj->hi', S, q[i])

        # RMSNorm par tête × porte sigmoïde g2
        var = sorties.pow(2).mean(-1, keepdim=True)
        normed = sorties * torch.rsqrt(var + self.eps) \
            * self.norm_weight.to(torch.float32)
        g2 = self.g_b(self.g_a(x)).to(torch.float32).reshape(t, self.nh, self.d)
        y = (normed * torch.sigmoid(g2)).reshape(t, self.d_inner)
        return (self.out_proj(y.to(x.dtype)),
                (new_cq, new_ck, new_cv, S))
