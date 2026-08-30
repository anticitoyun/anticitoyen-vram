"""The transformer itself: llama-family dense and MoE, placed across tiers.

Scope is deliberate. This covers the architecture that essentially every
open-weights model in the size range worth running on this rig uses -- RMSNorm,
RoPE, grouped-query attention, SwiGLU, and optionally a sparse MoE feed-forward
-- rather than trying to be a general model zoo. Llama, Mistral, Qwen2/3,
Mixtral and DeepSeek all fit it.

What is specific to this project is that a layer knows which device it runs on
and whether its weights are resident or streamed, and that the MoE block only
touches the experts the router selected, which is what makes host RAM a
sensible place to keep the other 120 of them.
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
    """One step of work, prefill or decode.

    ``seq_lens`` is the total context length of each sequence *after* the
    tokens in this batch are appended, which is what attention needs to know
    how much history to read.
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
        """Absolute position at which each sequence's query block starts."""
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
                # Part of this sequence is already in the cache -- a served
                # prefix, or an earlier chunk. Read it back and mask against
                # the query's absolute offset.
                kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                      batch.seq_lens[i], q.dtype)
            else:
                # Nothing prior: use the keys we just computed rather than
                # reading them back through the cache. That skips a
                # quantize/dequantize round trip on every prefill token, which
                # is both faster and slightly more accurate.
                kk, vv = k[start:end], v[start:end]
            kk = repeat_kv(kk, self.n_rep)
            vv = repeat_kv(vv, self.n_rep)
            out[start:end] = attention(q[start:end], kk, vv, True, self.scale,
                                       q_offset=offset)
            start = end
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

    def _decode(self, q, k, v, batch: ForwardBatch,
                cache: Optional[PagedKVCache], t: int) -> torch.Tensor:
        """Decode, and speculative verification, for the whole batch at once."""
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

        # Speculative verification: several query positions per sequence, each
        # attending to its own prefix. Still causal, still offset.
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
    """Sparse mixture of experts.

    Only the ``top_k`` experts a token routed to are evaluated, so the cost of
    a layer is independent of how many experts it owns. That is what lets 128
    experts sit in host RAM while the model still decodes at a usable rate:
    per token the machine reads 8 of them, not 128.
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
        # Group tokens by expert so each expert runs one batched GEMM rather
        # than one per token.
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
        # Attention and MLP need not run on the same device. When the MLP's
        # weights live in host RAM it is usually cheaper to compute them there
        # than to copy them across PCIe -- see PlannerOptions.host_exec.
        self.mlp_device = mlp_device or device

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), batch, cache)
        h = self.post_attention_layernorm(x)
        if self.mlp_device != self.device:
            # Only the hidden state crosses the bus: [tokens, hidden], a few
            # kilobytes per decoded token against gigabytes of weights.
            y = self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
        else:
            y = self.mlp(h)
        return x + y

    def prefetch(self) -> None:
        for m in self.modules():
            if isinstance(m, QuantLinear) and m.streamed is not None:
                m.prefetch()


class ACVRamModel(nn.Module):
    """The assembled model, with its layers spread across devices."""

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
        """Logits for the last token of every sequence.

        With ``return_hidden`` the normalised hidden states are returned
        instead, for every token rather than only the last -- that is what
        /v1/embeddings pools over. Skipping lm_head also skips the single
        most expensive GEMM in the model, so embedding a document costs
        noticeably less than generating from it.
        """
        idx = batch.tokens.to(self.embed_tokens.device)
        x = F.embedding(idx, self.embed_tokens).to(self.dtype)

        current = None
        for i, layer in enumerate(self.layers):
            if layer.device != current:
                x = x.to(layer.device, non_blocking=True)
                current = layer.device
            # Start the next layer's transfer before running this one, so a
            # host-resident layer's PCIe copy hides behind real work.
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch()
            x = layer(x, batch, self.caches.get(i))

        x = self.norm(x.to(self.norm.weight.device))
        if return_hidden:
            return x
        # Speculative verification and perplexity both need logits at more
        # than the final position, so which rows reach lm_head is a parameter.
        # It matters: lm_head is the single largest GEMM in the model, and
        # running it on every prefill token instead of one costs real time.
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
