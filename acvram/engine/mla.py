"""MLA — Multi-head Latent Attention (couches d'attention pleine de
« kimi-linear »), en forme absorbée, fidèle au graphe llama.cpp.

Particularités de Kimi-Linear : **aucun RoPE** (la position ne passe que par
la récurrence KDA des autres couches) ; le cache d'une séquence est le latent
compressé ``[t, kv_lora_rank + rope_dim]`` (576 octets ×2 par jeton et par
couche) — pas de K/V par tête, pas de cache paginé.

Forme absorbée : ``q_eff = [k_b(q_nope) | q_pe]`` s'apparie au latent caché
``[c | k_pe]`` ; la valeur est relue dans l'espace latent puis décompressée
par ``v_b``. Échelle 1/√(dim_qk_complet), comme llama.cpp (kq_scale_mla).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["MLAttention"]


class MLAttention(nn.Module):
    def __init__(self, q_proj: nn.Module, kv_a_proj: nn.Module,
                 o_proj: nn.Module,
                 kv_a_norm: torch.Tensor,          # [kv_lora_rank]
                 k_b: torch.Tensor,                # [heads, kv_lora_rank, qk_nope]
                 v_b: torch.Tensor,                # [heads, v_dim, kv_lora_rank]
                 num_heads: int, qk_nope: int, qk_rope: int,
                 kv_lora_rank: int, v_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.q_proj, self.kv_a_proj, self.o_proj = q_proj, kv_a_proj, o_proj
        self.kv_a_norm = nn.Parameter(kv_a_norm, requires_grad=False)
        self.k_b = nn.Parameter(k_b, requires_grad=False)
        self.v_b = nn.Parameter(v_b, requires_grad=False)
        self.nh = num_heads
        self.nope, self.rope = qk_nope, qk_rope
        self.rank, self.dv = kv_lora_rank, v_dim
        self.scale = (qk_nope + qk_rope) ** -0.5
        self.eps = eps

    def forward(self, x: torch.Tensor,
                cache: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, cache latent)."""
        t = x.shape[0]
        q = self.q_proj(x).reshape(t, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)

        kvp = self.kv_a_proj(x)                               # [t, rank+rope]
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        c32 = c.to(torch.float32)
        c = (c32 * torch.rsqrt(c32.pow(2).mean(-1, keepdim=True) + self.eps)
             ).to(x.dtype) * self.kv_a_norm

        # q absorbé : [t, nh, rank] = k_b [nh, rank, nope] @ q_nope [t, nh, nope]
        q_abs = torch.einsum('hrn,thn->thr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)              # [t, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                  # [t, rank+rope]

        cache = k_new if cache is None else torch.cat([cache, k_new], dim=0)
        total = cache.shape[0]

        scores = torch.einsum('thr,sr->ths', q_eff.to(torch.float32),
                              cache.to(torch.float32)) * self.scale
        if t > 1:                                             # masque causal
            passe = total - t
            pos_q = torch.arange(t, device=x.device).unsqueeze(-1) + passe
            pos_k = torch.arange(total, device=x.device)
            scores = scores.masked_fill(
                pos_k > pos_q.unsqueeze(1), float('-inf'))
        probs = scores.softmax(dim=-1)

        o_lat = torch.einsum('ths,sr->thr', probs,
                             cache[:, :self.rank].to(torch.float32))
        y = torch.einsum('hvr,thr->thv', self.v_b.to(torch.float32), o_lat)
        y = y.reshape(t, self.nh * self.dv).to(x.dtype)
        return self.o_proj(y), cache
