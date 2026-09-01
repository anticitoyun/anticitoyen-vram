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

__all__ = ["MLAttention", "MLA_BUCKET"]

# Godet de longueur du cache latent : partagé avec le chemin des graphes.
MLA_BUCKET = 1024


def _extension():
    import os
    if os.environ.get("ACVRAM_HYBRID_KERNELS", "1") == "0":
        return None
    from .. import kernels
    ext = kernels.get_extension()
    return ext if ext is not None and hasattr(ext, "mla_decode") else None


class MLAttention(nn.Module):
    def __init__(self, q_proj: nn.Module, kv_a_proj: nn.Module,
                 o_proj: nn.Module,
                 kv_a_norm: torch.Tensor,          # [kv_lora_rank]
                 k_b: torch.Tensor,                # [heads, kv_lora_rank, qk_nope]
                 v_b: torch.Tensor,                # [heads, v_dim, kv_lora_rank]
                 num_heads: int, qk_nope: int, qk_rope: int,
                 kv_lora_rank: int, v_dim: int,
                 eps: float = 1e-6,
                 q_a_proj: Optional[nn.Module] = None,
                 q_a_norm: Optional[torch.Tensor] = None,
                 q_b_proj: Optional[nn.Module] = None,
                 rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.q_proj, self.kv_a_proj, self.o_proj = q_proj, kv_a_proj, o_proj
        # DeepSeek-V2/GLM : q en bas rang (q_b(norm(q_a(x)))) et RoPE sur la
        # partie pe de q et de k (Kimi : ni l'un ni l'autre)
        self.q_a_proj, self.q_b_proj = q_a_proj, q_b_proj
        self.q_a_norm = nn.Parameter(q_a_norm, requires_grad=False) if q_a_norm is not None else None
        self.rope_emb = rope
        self.kv_a_norm = nn.Parameter(kv_a_norm, requires_grad=False)
        self.k_b = nn.Parameter(k_b, requires_grad=False)
        self.v_b = nn.Parameter(v_b, requires_grad=False)
        self.nh = num_heads
        self.nope, self.rope = qk_nope, qk_rope
        self.rank, self.dv = kv_lora_rank, v_dim
        self.scale = (qk_nope + qk_rope) ** -0.5
        self.eps = eps

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_a_proj is None:
            return self.q_proj(x)
        a = self.q_a_proj(x)
        a32 = a.to(torch.float32)
        a = (a32 * torch.rsqrt(a32.pow(2).mean(-1, keepdim=True) + self.eps)
             ).to(x.dtype) * self.q_a_norm
        return self.q_b_proj(a)

    def _rope(self, q_pe: torch.Tensor, k_pe: torch.Tensor,
              positions: torch.Tensor, max_pos: int):
        """RoPE (demi-rotation) sur les parties pe : q_pe [t, nh, r], k_pe [t, r]."""
        if self.rope_emb is None:
            return q_pe, k_pe
        # DeepSeek-V2/GLM : RoPE de type « norm » (paires adjacentes 2i, 2i+1),
        # pas la demi-rotation NEOX — llama.cpp le range hors LLAMA_ROPE_TYPE_NEOX
        cos, sin = self.rope_emb(positions, q_pe.device, q_pe.dtype, max_pos=max_pos)
        half = cos.shape[-1] // 2
        c = cos[..., :half].unsqueeze(1)                       # [t, 1, r/2]
        s = sin[..., :half].unsqueeze(1)

        def tourner(x: torch.Tensor) -> torch.Tensor:
            x2 = x.reshape(*x.shape[:-1], half, 2)
            x0, x1 = x2[..., 0], x2[..., 1]
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
            return torch.stack((y0, y1), dim=-1).reshape(x.shape)
        return tourner(q_pe), tourner(k_pe.unsqueeze(1)).squeeze(1)

    def forward(self, x: torch.Tensor,
                cache: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, cache latent)."""
        t = x.shape[0]
        q = self._q(x).reshape(t, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)

        kvp = self.kv_a_proj(x)                               # [t, rank+rope]
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        if self.rope_emb is not None:
            passe0 = 0 if cache is None else cache.shape[0]
            pos = torch.arange(passe0, passe0 + t, device=x.device)
            q_pe, k_pe = self._rope(q_pe, k_pe, pos, passe0 + t + 1)
        c32 = c.to(torch.float32)
        c = (c32 * torch.rsqrt(c32.pow(2).mean(-1, keepdim=True) + self.eps)
             ).to(x.dtype) * self.kv_a_norm

        # q absorbé : [t, nh, rank] = k_b [nh, rank, nope] @ q_nope [t, nh, nope]
        q_abs = torch.einsum('hrn,thn->thr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)              # [t, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                  # [t, rank+rope]

        cache = k_new if cache is None else torch.cat([cache, k_new], dim=0)
        total = cache.shape[0]

        if t == 1:
            # Décodage : même formulation en godet que ``decode_static`` (le
            # chemin des graphes), pour que les deux arrondissent pareil —
            # une GEMM sur N colonnes n'accumule pas comme sur 1024.
            bucket = -(-total // MLA_BUCKET) * MLA_BUCKET
            C = torch.zeros(bucket, cache.shape[1], dtype=cache.dtype,
                            device=cache.device)
            C[:total] = cache
            pos = torch.arange(bucket, device=x.device)
            masque = pos > (total - 1)
        else:
            # prefill : par tranches de requêtes, sinon les scores
            # [t, nh, total] fp32 pèsent des gigaoctets (2 Go à 4k jetons)
            passe = total - t
            C32 = cache.to(torch.float32)
            V32 = C32[:, :self.rank]
            pos_k = torch.arange(total, device=x.device)
            morceaux = []
            for d0 in range(0, t, 256):
                d1 = min(t, d0 + 256)
                sc = torch.einsum('thr,sr->ths', q_eff[d0:d1].to(torch.float32),
                                  C32) * self.scale
                pos_q = torch.arange(d0, d1, device=x.device).unsqueeze(-1) + passe
                sc = sc.masked_fill(pos_k > pos_q.unsqueeze(1), float('-inf'))
                morceaux.append(torch.einsum('ths,sr->thr', sc.softmax(dim=-1), V32))
            o_lat = torch.cat(morceaux)
            y = torch.einsum('hvr,thr->thv', self.v_b.to(torch.float32), o_lat)
            y = y.reshape(t, self.nh * self.dv).to(x.dtype)
            return self.o_proj(y), cache
        scores = torch.einsum('thr,sr->ths', q_eff.to(torch.float32),
                              C.to(torch.float32)) * self.scale
        scores = scores.masked_fill(masque, float('-inf'))
        probs = scores.softmax(dim=-1)

        o_lat = torch.einsum('ths,sr->thr', probs,
                             C[:, :self.rank].to(torch.float32))
        y = torch.einsum('hvr,thr->thv', self.v_b.to(torch.float32), o_lat)
        y = y.reshape(t, self.nh * self.dv).to(x.dtype)
        return self.o_proj(y), cache

    def fuse_projections(self) -> bool:
        from .layers import stack_int8_linears
        if self.q_a_proj is not None:
            self.q_kv = None
            return False
        self.q_kv = stack_int8_linears([self.q_proj, self.kv_a_proj])
        return self.q_kv is not None

    # -- chemin à formes fixes (graphes CUDA) --------------------------------
    def new_static(self, device: torch.device, max_len: int,
                   dtype: torch.dtype) -> dict:
        return {"cache": torch.zeros(max_len, self.rank + self.rope,
                                     dtype=dtype, device=device),
                "len": torch.zeros((), dtype=torch.long, device=device),
                # scores de travail du noyau fusionné [nh, max_len]
                "scores": torch.zeros(self.nh, max_len, dtype=torch.float32,
                                      device=device)}

    @staticmethod
    def static_load(st: dict, etat) -> None:
        if etat is None:
            st["len"].zero_()
            return
        n = etat.shape[0]
        st["cache"][:n].copy_(etat)
        st["len"].fill_(n)

    @staticmethod
    def static_export(st: dict):
        n = int(st["len"].item())
        return st["cache"][:n].clone()

    def decode_static(self, x: torch.Tensor, st: dict, bucket: int
                      ) -> torch.Tensor:
        """Un jeton, une séquence ; attention sur ``cache[:bucket]`` masquée
        au-delà de ``len`` ; écrit le latent à la ligne ``len`` puis avance."""
        # mêmes formulations que ``forward`` (t = 1), pour arrondir pareil
        if getattr(self, "q_kv", None) is not None:
            qkv = self.q_kv(x)
            nq = self.nh * (self.nope + self.rope)
            q = qkv[:, :nq].reshape(1, self.nh, self.nope + self.rope)
            kvp = qkv[:, nq:]
        else:
            q = self._q(x).reshape(1, self.nh, self.nope + self.rope)
            kvp = self.kv_a_proj(x)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        if self.rope_emb is not None:
            c0, k_pe0 = kvp.split([self.rank, self.rope], dim=-1)
            q_pe, k_pe0 = self._rope(q_pe, k_pe0, st["len"].view(1), bucket + 1)
            kvp = torch.cat([c0, k_pe0], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        c32 = c.to(torch.float32)
        c = (c32 * torch.rsqrt(c32.pow(2).mean(-1, keepdim=True) + self.eps)
             ).to(x.dtype) * self.kv_a_norm
        q_abs = torch.einsum('hrn,thn->thr', self.k_b.to(x.dtype), q_nope)
        q_eff = torch.cat([q_abs, q_pe], dim=-1)             # [1, nh, rank+rope]
        k_new = torch.cat([c, k_pe], dim=-1)                 # [1, rank+rope]
        cache = st["cache"]
        cache.index_copy_(0, st["len"].view(1), k_new)
        ext = _extension() if x.is_cuda else None
        if ext is not None:
            o_lat = ext.mla_decode(q_eff.to(torch.float32)[0].contiguous(),
                                   cache, st["len"], st["scores"], bucket,
                                   self.rank, self.scale)        # [nh, rank]
            y = torch.einsum('hvr,hr->hv', self.v_b.to(torch.float32), o_lat)
            st["len"].add_(1)
            return self.o_proj(y.reshape(1, self.nh * self.dv).to(x.dtype))
        C = cache[:bucket]
        scores = torch.einsum('thr,sr->ths', q_eff.to(torch.float32),
                              C.to(torch.float32)) * self.scale
        pos = torch.arange(bucket, device=x.device)
        scores = scores.masked_fill(pos > st["len"], float('-inf'))
        probs = scores.softmax(dim=-1)
        o_lat = torch.einsum('ths,sr->thr', probs,
                             C[:, :self.rank].to(torch.float32))
        y = torch.einsum('hvr,thr->thv', self.v_b.to(torch.float32), o_lat)
        st["len"].add_(1)
        return self.o_proj(y.reshape(1, self.nh * self.dv).to(x.dtype))
