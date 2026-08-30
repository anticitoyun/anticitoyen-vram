"""Le transformeur lui-même : famille llama, dense et à mélange d'experts,
réparti sur les étages de mémoire.

Le périmètre est délibéré. Il couvre l'architecture qu'emploie à peu près tout
modèle à poids ouverts de la gamme de tailles qui vaut la peine d'être exécutée
sur cette machine — RMSNorm, RoPE, attention à requêtes groupées, SwiGLU, et
optionnellement un bloc à mélange d'experts creux — plutôt que d'essayer d'être
une ménagerie universelle. Llama, Mistral, Qwen2/3, Mixtral et DeepSeek y
entrent tous.

Ce qui est propre à ce projet, c'est qu'une couche sait sur quel appareil elle
s'exécute et si ses poids sont résidents ou transférés, et que le bloc à mélange
d'experts ne touche que les experts choisis par le routeur — ce qui fait de la
mémoire vive un endroit raisonnable pour garder les 120 autres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory.kvcache import PagedKVCache
from .config import ModelSpec
from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, apply_rope,
                     attention, batched_decode_attention, repeat_kv)

__all__ = ["Attention", "MLP", "MoEBlock", "DecoderLayer", "ACVRamModel",
           "ForwardBatch"]


@dataclass
class ForwardBatch:
    """Une étape de travail, prefill ou décodage.

    ``seq_lens`` est la longueur de contexte totale de chaque séquence *après*
    ajout des jetons de ce lot : c'est ce dont l'attention a besoin pour savoir
    quelle quantité d'histoire lire.
    """

    tokens: torch.Tensor              # [total_tokens] flattened across sequences
    positions: torch.Tensor           # [total_tokens]
    seq_lens: list[int]
    query_lens: list[int]
    block_tables: list[torch.Tensor]
    slot_mapping: torch.Tensor        # [total_tokens]
    is_prefill: bool

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def is_decode(self) -> bool:
        return not self.is_prefill

    @property
    def q_offsets(self) -> list[int]:
        """Position absolue à laquelle commence le bloc de requêtes de chaque séquence."""
        return [s - q for s, q in zip(self.seq_lens, self.query_lens)]

    def last_token_indices(self) -> torch.Tensor:
        out, pos = [], 0
        for qlen in self.query_lens:
            pos += qlen
            out.append(pos - 1)
        return torch.tensor(out, dtype=torch.long)

    def all_token_indices(self) -> torch.Tensor:
        return torch.arange(self.tokens.shape[0], dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, spec: ModelSpec, q: QuantLinear, k: QuantLinear,
                 v: QuantLinear, o: QuantLinear, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.n_rep = self.n_heads // max(1, self.n_kv_heads)
        self.scale = self.head_dim ** -0.5
        self.rope = rope

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.n_kv_heads, self.head_dim)

        cos, sin = self.rope(batch.positions.to(x.device), x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            cache.write(batch.slot_mapping.to(x.device), k, v)

        if batch.is_decode:
            return self._decode(q, k, v, batch, cache, t)
        return self._prefill(q, k, v, batch, cache, t)

    def _prefill(self, q, k, v, batch: ForwardBatch,
                 cache: Optional[PagedKVCache], t: int) -> torch.Tensor:
        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            if cache is not None and offset > 0:
                # Une partie de cette séquence est déjà en cache : un préfixe
                # servi, ou un morceau antérieur. On la relit et on masque selon
                # le décalage absolu de la requête.
                kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                      batch.seq_lens[i], q.dtype)
            else:
                # Rien avant : on utilise les clés qu'on vient de calculer
                # plutôt que de les relire par le cache. Cela évite un
                # aller-retour de quantification sur chaque jeton de prefill, ce
                # qui est à la fois plus rapide et un peu plus précis.
                kk, vv = k[start:end], v[start:end]
            kk = repeat_kv(kk, self.n_rep)
            vv = repeat_kv(vv, self.n_rep)
            out[start:end] = attention(q[start:end], kk, vv, True, self.scale,
                                       q_offset=offset)
            start = end
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

    def _decode(self, q, k, v, batch: ForwardBatch,
                cache: Optional[PagedKVCache], t: int) -> torch.Tensor:
        """Décodage, et vérification spéculative, pour tout le lot d'un coup."""
        if cache is None:
            return self._prefill(q, k, v, batch, cache, t)

        keys, values = [], []
        for i in range(batch.batch_size):
            kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                  batch.seq_lens[i], q.dtype)
            keys.append(kk)
            values.append(vv)

        if all(ql == 1 for ql in batch.query_lens):
            out = batched_decode_attention(q, keys, values, self.n_rep, self.scale)
            out = out.reshape(t, self.n_heads * self.head_dim)
            return self.o_proj(out)

        # Vérification spéculative : plusieurs positions de requête par
        # séquence, chacune attendant sur son propre préfixe. Toujours causal,
        # toujours décalé.
        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            out[start:end] = attention(
                q[start:end], repeat_kv(keys[i], self.n_rep),
                repeat_kv(values[i], self.n_rep), True, self.scale,
                q_offset=offset)
            start = end
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, gate: QuantLinear, up: QuantLinear, down: QuantLinear) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    """Mélange d'experts creux.

    Seuls les ``top_k`` experts vers lesquels un jeton a été routé sont évalués :
    le coût d'une couche est donc indépendant du nombre d'experts qu'elle
    possède. C'est ce qui permet à 128 experts de résider en mémoire vive pendant
    que le modèle décode à un rythme exploitable — par jeton, la machine en lit
    8, pas 128.
    """

    def __init__(self, router: QuantLinear, experts: list[MLP], top_k: int,
                 shared: Optional[MLP] = None,
                 norm_topk_prob: bool = True) -> None:
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.shared = shared
        self.norm_topk_prob = norm_topk_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h = x.shape
        logits = self.router(x.to(torch.float32))
        weights = F.softmax(logits, dim=-1)
        topw, topi = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw.to(x.dtype)

        out = torch.zeros_like(x)
        # On regroupe les jetons par expert, pour que chaque expert fasse un
        # seul produit matriciel par lot au lieu d'un par jeton.
        flat_expert = topi.reshape(-1)
        flat_weight = topw.reshape(-1)
        flat_token = torch.arange(t, device=x.device).repeat_interleave(self.top_k)
        for e in flat_expert.unique().tolist():
            sel = flat_expert == e
            tok = flat_token[sel]
            y = self.experts[e](x[tok])
            out.index_add_(0, tok, y * flat_weight[sel].unsqueeze(-1))
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def prefetch(self) -> None:
        for expert in self.experts:
            for lin in (expert.gate_proj, expert.up_proj, expert.down_proj):
                lin.prefetch()


class DecoderLayer(nn.Module):
    def __init__(self, index: int, attn: Attention, mlp: nn.Module,
                 input_norm: RMSNorm, post_norm: RMSNorm, device: torch.device,
                 mlp_device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.index = index
        self.self_attn = attn
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_norm
        self.device = device
        # L'attention et le MLP n'ont pas à s'exécuter sur le même appareil.
        # Quand les poids du MLP résident en mémoire vive, il est en général
        # moins coûteux de les y calculer que de les copier par le PCIe — voir
        # PlannerOptions.host_exec.
        self.mlp_device = mlp_device or device

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), batch, cache)
        h = self.post_attention_layernorm(x)
        if self.mlp_device != self.device:
            # Seul l'état caché traverse le bus : [jetons, dimension], quelques
            # kilooctets par jeton décodé face à des gigaoctets de poids.
            y = self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
        else:
            y = self.mlp(h)
        return x + y

    def prefetch(self) -> None:
        for m in self.modules():
            if isinstance(m, QuantLinear) and m.streamed is not None:
                m.prefetch()


class ACVRamModel(nn.Module):
    """Le modèle assemblé, ses couches réparties sur plusieurs appareils."""

    def __init__(self, spec: ModelSpec, embed: torch.Tensor,
                 layers: list[DecoderLayer], norm: RMSNorm,
                 lm_head: QuantLinear, caches: dict[int, PagedKVCache],
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = embed          # kept as a plain tensor: it is a gather
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head
        self.caches = caches
        self.dtype = dtype

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, return_hidden: bool = False,
                logits_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits du dernier jeton de chaque séquence.

        Avec ``return_hidden``, ce sont les états cachés normalisés qui sont
        rendus, pour tous les jetons et non seulement le dernier — c'est sur eux
        que /v1/embeddings fait sa moyenne. Sauter lm_head évite aussi le produit
        matriciel le plus coûteux du modèle : plonger un document coûte donc
        nettement moins que d'engendrer à partir de lui.
        """
        idx = batch.tokens.to(self.embed_tokens.device)
        x = F.embedding(idx, self.embed_tokens).to(self.dtype)

        current = None
        for i, layer in enumerate(self.layers):
            if layer.device != current:
                x = x.to(layer.device, non_blocking=True)
                current = layer.device
            # On lance le transfert de la couche suivante avant d'exécuter
            # celle-ci, pour que la copie PCIe d'une couche résidant en mémoire
            # vive se cache derrière du vrai travail.
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch()
            x = layer(x, batch, self.caches.get(i))

        x = self.norm(x.to(self.norm.weight.device))
        if return_hidden:
            return x
        # La vérification spéculative et la perplexité ont toutes deux besoin
        # de logits ailleurs qu'à la position finale : les lignes qui atteignent
        # lm_head sont donc un paramètre. Cela compte, car lm_head est le plus
        # gros produit matriciel du modèle, et l'exécuter sur chaque jeton de
        # prefill au lieu d'un seul coûte du temps réel.
        idx = (batch.last_token_indices() if logits_positions is None
               else logits_positions)
        x = x[idx.to(x.device)]
        head_dev = getattr(self.lm_head.qweight, "qweight", None)
        target = head_dev.device if head_dev is not None else x.device
        return self.lm_head(x.to(target))

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.numel() * self.embed_tokens.element_size()
        for m in self.modules():
            if isinstance(m, QuantLinear):
                total += m.nbytes
        return total
