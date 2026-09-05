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

import os

import torch
import torch.nn as nn

__all__ = ["MLAttention", "MLA_BUCKET", "godet_mla"]

# Godet de longueur du cache latent : partagé avec le chemin des graphes.
# L'attention MLA balaie tout le godet, pas seulement les positions écrites :
# à 1024, une séquence de 264 jetons en paie quatre fois trop. Un godet plus
# fin coûte davantage de graphes capturés (un par palier), mais chaque pas
# lit moins. Réglable pour mesurer l'arbitrage.
MLA_BUCKET = int(os.environ.get("ACVRAM_MLA_BUCKET", "128"))


def godet_mla(longueur: int) -> int:
    """Palier de cache latent couvrant ``longueur``, en puissances de deux.

    L'attention MLA balaie tout le godet, pas seulement les positions écrites :
    à godet fixe de 1024, une séquence de 264 jetons payait quatre fois trop.
    Un godet fixe et fin coûterait en revanche un graphe par palier — 256
    paliers pour un contexte de 32 768, bien au-delà de ce qu'on capture. Des
    paliers doublants tiennent les deux bouts : une courte séquence ne lit que
    ce qu'il lui faut, et le contexte entier ne demande que neuf paliers.
    """
    n = MLA_BUCKET
    while n < longueur:
        n *= 2
    return n


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
        self.q_kv = None            # q_proj et kv_a_proj empilées
        self.qa_kv = None           # q_a_proj et kv_a_proj empilées (bas rang)
        self.q_a_taille = 0

    def _proj_entree(self, x: torch.Tensor):
        """Première projection de q et latent kv : une GEMV quand les deux
        lisent la même entrée, deux sinon.

        C'est le cas de toutes les variantes DeepSeek-V2 et GLM : ``q_a_proj``
        et ``kv_a_proj`` partent l'une comme l'autre de l'état caché.
        """
        if self.qa_kv is not None:
            g = self.qa_kv(x)
            return g[:, :self.q_a_taille], g[:, self.q_a_taille:]
        if self.q_kv is not None:
            g = self.q_kv(x)
            nq = self.nh * (self.nope + self.rope)
            return g[:, :nq], g[:, nq:]
        prem = self.q_proj(x) if self.q_a_proj is None else self.q_a_proj(x)
        return prem, self.kv_a_proj(x)

    def _norme(self, t: torch.Tensor, poids: torch.Tensor) -> torch.Tensor:
        """RMSNorm : un lancement quand le noyau est là, sept sinon.

        La formulation PyTorch — conversion, carré, moyenne, racine inverse,
        deux multiplications, reconversion — coûte sept noyaux pour une poignée
        de milliers d'éléments. Sur 47 couches et deux normalisations par
        couche, c'est plus de la moitié des noyaux élémentaires d'un jeton.
        """
        ext = _extension() if t.is_cuda else None
        if os.environ.get("ACVRAM_MLA_NORME_NOYAU") == "0":   # témoin de mesure
            ext = None
        if (ext is not None and hasattr(ext, "rmsnorm_bf16")
                and t.dtype == torch.bfloat16 and poids.dtype == torch.bfloat16):
            return ext.rmsnorm_bf16(t, poids, self.eps)[0]
        t32 = t.to(torch.float32)
        return (t32 * torch.rsqrt(t32.pow(2).mean(-1, keepdim=True) + self.eps)
                ).to(t.dtype) * poids

    def _q_depuis(self, prem: torch.Tensor) -> torch.Tensor:
        """De la sortie de la première projection au q complet."""
        if self.q_a_proj is None:
            return prem
        return self.q_b_proj(self._norme(prem, self.q_a_norm))

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_depuis(self._proj_entree(x)[0])

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
        # q et k tournent ensemble : la rotation est point à point et ne
        # dépend pas du nombre de têtes, si bien que les traiter d'un bloc
        # rend exactement les mêmes bits pour moitié moins de lancements.
        ensemble = tourner(torch.cat([q_pe, k_pe.unsqueeze(1)], dim=1))
        return ensemble[:, :-1], ensemble[:, -1]

    def forward(self, x: torch.Tensor,
                cache: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` vaut [t, hidden] pour UNE séquence ; rend (y, cache latent)."""
        t = x.shape[0]
        prem, kvp = self._proj_entree(x)                      # kvp [t, rank+rope]
        q = self._q_depuis(prem).reshape(t, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        if self.rope_emb is not None:
            passe0 = 0 if cache is None else cache.shape[0]
            pos = torch.arange(passe0, passe0 + t, device=x.device)
            q_pe, k_pe = self._rope(q_pe, k_pe, pos, passe0 + t + 1)
        c = self._norme(c, self.kv_a_norm)

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
            bucket = godet_mla(total)
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
        """Une GEMV au lieu de deux à l'entrée de l'attention.

        Les modèles à q de bas rang (DeepSeek-V2, GLM-4.x) étaient exclus :
        seule la paire ``q_proj``/``kv_a_proj`` était traitée, et en INT8
        seulement. Or sur GLM-4.7-Flash, ``q_a_proj`` [768, 2048] et
        ``kv_a_proj`` [576, 2048] lisent toutes deux l'état caché — 47 couches
        qui lançaient chacune une GEMV de trop.
        """
        from .layers import stack_int8_linears, stack_nvfp4_linears

        def empiler(lins):
            return stack_int8_linears(lins) or stack_nvfp4_linears(lins)

        self.q_kv = self.qa_kv = None
        if self.q_a_proj is not None:
            self.qa_kv = empiler([self.q_a_proj, self.kv_a_proj])
            if self.qa_kv is not None:
                self.q_a_taille = int(self.q_a_proj.qweight.shape[0])
            return self.qa_kv is not None
        self.q_kv = empiler([self.q_proj, self.kv_a_proj])
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
        prem, kvp = self._proj_entree(x)
        q = self._q_depuis(prem).reshape(1, self.nh, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        if self.rope_emb is not None:
            c0, k_pe0 = kvp.split([self.rank, self.rope], dim=-1)
            q_pe, k_pe0 = self._rope(q_pe, k_pe0, st["len"].view(1), bucket + 1)
            kvp = torch.cat([c0, k_pe0], dim=-1)
        c, k_pe = kvp.split([self.rank, self.rope], dim=-1)
        c = self._norme(c, self.kv_a_norm)
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
