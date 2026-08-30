"""Model description, derived from a Hugging Face ``config.json``.

Only what the planner and the runtime need: shapes, parameter counts per
layer group, and whether the layer is a sparse MoE (which changes the cost of
putting it in host RAM by an order of magnitude).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

__all__ = ["ModelSpec", "LayerSpec", "load_model_spec"]


@dataclass
class LayerSpec:
    """One transformer block, split into the tensors the planner moves."""

    index: int
    attn_params: int
    mlp_params: int
    norm_params: int
    is_moe: bool = False
    n_experts: int = 0
    n_experts_active: int = 0
    shared_expert_params: int = 0

    @property
    def total_params(self) -> int:
        return self.attn_params + self.mlp_params + self.norm_params

    @property
    def active_params(self) -> int:
        """Parameters actually read for a single token.

        For a dense block this is everything. For an MoE block only the router,
        the shared expert and ``n_experts_active`` of the experts are touched,
        which is why a 235B MoE streams from host RAM at a workable speed while
        a 70B dense model does not.
        """
        if not self.is_moe or self.n_experts == 0:
            return self.total_params
        per_expert = (self.mlp_params - self.shared_expert_params) / self.n_experts
        return int(self.attn_params + self.norm_params + self.shared_expert_params
                   + per_expert * self.n_experts_active)

    @property
    def activation_ratio(self) -> float:
        return self.active_params / max(1, self.total_params)


@dataclass
class ModelSpec:
    name: str
    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    tie_word_embeddings: bool = False
    head_dim: int = 0
    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    first_k_dense_replace: int = 0
    torch_dtype: str = "bfloat16"
    layers: list[LayerSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.head_dim:
            self.head_dim = self.hidden_size // max(1, self.num_attention_heads)
        if not self.layers:
            self.layers = [self._build_layer(i) for i in range(self.num_layers)]

    def _build_layer(self, i: int) -> LayerSpec:
        h, hd = self.hidden_size, self.head_dim
        q = h * self.num_attention_heads * hd
        kv = 2 * h * self.num_key_value_heads * hd
        o = self.num_attention_heads * hd * h
        attn = q + kv + o
        norm = 2 * h

        is_moe = self.num_experts > 0 and i >= self.first_k_dense_replace
        if is_moe:
            inter = self.moe_intermediate_size or self.intermediate_size
            per_expert = 3 * h * inter
            router = h * self.num_experts
            shared = 3 * h * self.shared_expert_intermediate_size \
                if self.shared_expert_intermediate_size else 0
            mlp = per_expert * self.num_experts + router + shared
            return LayerSpec(i, attn, mlp, norm, True, self.num_experts,
                             self.num_experts_per_tok, shared + router)
        mlp = 3 * h * self.intermediate_size
        return LayerSpec(i, attn, mlp, norm)

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

    @property
    def total_params(self) -> int:
        return (self.embed_params + self.lm_head_params + self.hidden_size
                + sum(l.total_params for l in self.layers))

    @property
    def active_params(self) -> int:
        return (self.embed_params + self.lm_head_params
                + sum(l.active_params for l in self.layers))

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    def kv_bytes_per_token(self, kv_bits: int = 8) -> int:
        """Bytes of KV cache for one token across all layers.

        GQA is already accounted for: only ``num_key_value_heads`` are stored.
        """
        per_layer = 2 * self.num_key_value_heads * self.head_dim * kv_bits / 8
        # grouped scales for quantized KV: one fp16 per head per token per kv
        overhead = 0.0 if kv_bits >= 16 else 2 * self.num_key_value_heads * 2
        return int((per_layer + overhead) * self.num_layers)

    def summary(self) -> str:
        b = self.total_params / 1e9
        a = self.active_params / 1e9
        moe = f", MoE {self.num_experts}x top-{self.num_experts_per_tok}" if self.is_moe else ""
        return (f"{self.name}: {self.architecture}, {self.num_layers} layers, "
                f"h={self.hidden_size}, {b:.1f}B params "
                f"({a:.1f}B active/token){moe}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        d.pop("layers", None)
        d["total_params"] = self.total_params
        d["active_params"] = self.active_params
        return d


_ARCH_ALIASES = {
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",
    "Qwen3ForCausalLM": "llama",
    "Qwen2MoeForCausalLM": "moe",
    "Qwen3MoeForCausalLM": "moe",
    "MixtralForCausalLM": "moe",
    "DeepseekV2ForCausalLM": "moe",
    "DeepseekV3ForCausalLM": "moe",
    "GemmaForCausalLM": "llama",
    "Gemma2ForCausalLM": "llama",
    "Phi3ForCausalLM": "llama",
}


def load_model_spec(path: str, name: Optional[str] = None) -> ModelSpec:
    """Read a HF model directory (or a bare ``config.json``) into a ModelSpec."""
    cfg_path = path if path.endswith(".json") else os.path.join(path, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = json.load(fh)

    # Some configs nest the language model (VLMs).
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = {**cfg, **cfg["text_config"]}

    archs = cfg.get("architectures") or ["LlamaForCausalLM"]
    arch = _ARCH_ALIASES.get(archs[0], "llama")

    n_heads = cfg.get("num_attention_heads", 32)
    spec = ModelSpec(
        name=name or cfg.get("_name_or_path") or os.path.basename(os.path.abspath(path)),
        architecture=arch,
        hidden_size=cfg.get("hidden_size", 4096),
        intermediate_size=cfg.get("intermediate_size", 11008),
        num_layers=cfg.get("num_hidden_layers", 32),
        num_attention_heads=n_heads,
        num_key_value_heads=cfg.get("num_key_value_heads", n_heads),
        vocab_size=cfg.get("vocab_size", 32000),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_scaling=cfg.get("rope_scaling"),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        head_dim=cfg.get("head_dim", 0),
        num_experts=cfg.get("num_experts") or cfg.get("num_local_experts")
        or cfg.get("n_routed_experts") or 0,
        num_experts_per_tok=cfg.get("num_experts_per_tok") or cfg.get("top_k") or 0,
        moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
        shared_expert_intermediate_size=cfg.get("shared_expert_intermediate_size", 0),
        first_k_dense_replace=cfg.get("first_k_dense_replace", 0),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
        raw=cfg,
    )
    return spec
