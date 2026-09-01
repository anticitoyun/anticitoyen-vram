"""Description d'un modèle, dérivée d'un ``config.json`` Hugging Face.

Uniquement ce dont le planificateur et l'exécution ont besoin : les formes, le
nombre de paramètres par groupe de couches, et le fait qu'une couche soit ou
non à mélange d'experts creux — ce qui change d'un ordre de grandeur le coût de
la placer en mémoire vive.
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
    """Un bloc de transformeur, découpé selon les tenseurs que déplace le planificateur."""

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
        """Paramètres réellement lus pour un seul jeton.

        Pour un bloc dense, c'est la totalité. Pour un bloc à mélange d'experts,
        seuls le routeur, l'expert partagé et ``n_experts_active`` experts sont
        touchés — et c'est pourquoi un MoE de 235 milliards de paramètres se
        transfère depuis la mémoire vive à une vitesse exploitable, là où un
        modèle dense de 70 milliards ne le peut pas.
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
    # Jetons d'arret. Sans eux le moteur ne s'arrete jamais de lui-meme et
    # rend toujours max_tokens jetons, en repartant en roue libre apres la
    # reponse.
    eos_token_id: list[int] = field(default_factory=list)
    bos_token_id: Optional[int] = None
    # Hybrides à récurrence linéaire (qwen3-next) : type de chaque couche et
    # géométrie de la partie Gated DeltaNet. Vide = transformeur pur.
    layer_types: list[str] = field(default_factory=list)
    linear_num_value_heads: int = 0
    linear_num_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4
    rotary_dim: Optional[int] = None      # RoPE partiel (None = tête entière)
    attn_output_gate: bool = False
    # gemma4 : couches locales (fenêtre) / globales (têtes plus larges, RoPE
    # proportionnel), v normalisé, k = v global, softcap final
    sliding_window: int = 0
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    rope_theta_swa: float = 0.0
    partial_rotary_factor_full: float = 1.0
    final_logit_softcapping: float = 0.0
    hidden_activation: str = "silu"
    attention_k_eq_v: bool = False
    # granite : multiplicateurs scalaires (attention, plongement, résidu, logits)
    attention_multiplier: Optional[float] = None
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    # kimi-linear : KDA + MLA + routeur DeepSeek
    model_type: str = ""
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    router_scoring: str = "softmax"
    routed_scaling_factor: float = 1.0
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
        """Octets de cache KV pour un jeton, toutes couches confondues.

        L'attention à requêtes groupées est déjà prise en compte : seules
        ``num_key_value_heads`` têtes sont stockées.
        """
        per_layer = 2 * self.num_key_value_heads * self.head_dim * kv_bits / 8
        # échelles groupées du KV quantifié : un fp16 par tête, par jeton, par kv
        overhead = 0.0 if kv_bits >= 16 else 2 * self.num_key_value_heads * 2
        return int((per_layer + overhead) * self.num_layers)

    def summary(self) -> str:
        b = self.total_params / 1e9
        a = self.active_params / 1e9
        moe = (f", MoE {self.num_experts} experts, top-{self.num_experts_per_tok}"
               if self.is_moe else "")
        return (f"{self.name} : {self.architecture}, {self.num_layers} couches, "
                f"h={self.hidden_size}, {b:.1f} G parametres "
                f"({a:.1f} G actifs par jeton){moe}")

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
    "GraniteForCausalLM": "llama",
    "Gemma4ForCausalLM": "llama",
    "Gemma4ForConditionalGeneration": "llama",
    # vision-langage (partie texte seule)
    "Qwen2VLForConditionalGeneration": "llama",
    "Qwen2_5_VLForConditionalGeneration": "llama",
    "Qwen3VLForConditionalGeneration": "llama",
    "Qwen3VLMoeForConditionalGeneration": "moe",
    "Qwen3_5ForConditionalGeneration": "llama",
    "Qwen3_5MoeForConditionalGeneration": "moe",
}


def _as_id_list(v: Any) -> list[int]:
    """``eos_token_id`` vaut tantot un entier, tantot une liste."""
    if isinstance(v, int):
        return [v]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v if isinstance(x, int)]
    return []


def load_model_spec(path: str, name: Optional[str] = None) -> ModelSpec:
    """Lit un répertoire de modèle Hugging Face, ou un simple ``config.json``."""
    cfg_path = path if path.endswith(".json") else os.path.join(path, "config.json")
    if not path.endswith(".json") and not os.path.isfile(cfg_path):
        # Un point de contrôle GGUF porte sa configuration dans son en-tête.
        from ..quant.gguf import GGUFFile, is_gguf
        if is_gguf(path):
            g = GGUFFile(path)
            g.check_executable()
            cfg = g.hf_config()
        else:
            raise FileNotFoundError(f"ni config.json ni .gguf sous {path}")
    else:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

    # Certaines configurations imbriquent le modèle de langage (modèles visuels).
    if "text_config" in cfg and "hidden_size" not in cfg:
        cfg = {**cfg, **cfg["text_config"]}

    # generation_config.json fait autorite sur les jetons d'arret : Qwen y
    # declare <|im_end|>, absent du eos_token_id de config.json sur certains
    # points de controle.
    gen_path = (os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                             "generation_config.json")
                if os.path.isfile(cfg_path) else "")
    if os.path.isfile(gen_path):
        try:
            with open(gen_path, "r", encoding="utf-8") as fh:
                gen = json.load(fh)
            merged = _as_id_list(cfg.get("eos_token_id")) + \
                _as_id_list(gen.get("eos_token_id"))
            if merged:
                cfg = {**cfg, "eos_token_id": sorted(set(merged))}
        except (OSError, json.JSONDecodeError):
            pass

    archs = cfg.get("architectures") or ["LlamaForCausalLM"]
    mt = str(cfg.get("model_type", ""))
    if mt in ("gemma4", "gemma4_text"):
        rp = cfg.get("rope_parameters") or {}
        full, swa = rp.get("full_attention", {}), rp.get("sliding_attention", {})
        cfg = {**cfg,
               "rope_theta": float(full.get("rope_theta", cfg.get("rope_theta", 1e6))),
               "rope_theta_swa": float(swa.get("rope_theta", cfg.get("rope_theta_swa", 1e4))),
               "partial_rotary_factor_full": float(full.get("partial_rotary_factor",
                                                             cfg.get("partial_rotary_factor_full", 1.0))),
               "embedding_multiplier": float(cfg["hidden_size"]) ** 0.5,
               "attention_multiplier": 1.0}
    # qwen3_next est désormais exécutable (couches Gated DeltaNet) quand la
    # configuration porte nos champs layer_types/linear_* ; les autres
    # hybrides restent refusés.
    if mt in ("qwen3_next", "kimi_linear", "qwen3_5", "qwen3_5_text",
              "qwen3_5_moe", "qwen3_5_moe_text") \
            and cfg.get("linear_num_value_heads"):
        if cfg.get("rotary_dim") is None and cfg.get("partial_rotary_factor"):
            hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
            cfg = {**cfg, "rotary_dim": int(hd * float(cfg["partial_rotary_factor"]))}
    elif (mt in ("kimi_linear", "nemotron_h",
               "falcon_h1", "lfm2_moe", "mamba", "mamba2", "jamba")
            or "linear_attn" in json.dumps(cfg.get("layer_types", ""))):
        raise ValueError(
            f"architecture « {archs[0]} » (model_type={mt}) : recurrence "
            f"lineaire ou hybride SSM — le moteur acvram est un transformeur "
            f"pur et ne peut pas l'executer.")
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
        layer_types=list(cfg.get("layer_types") or []),
        linear_num_value_heads=int(cfg.get("linear_num_value_heads") or 0),
        linear_num_key_heads=int(cfg.get("linear_num_key_heads") or 0),
        linear_key_head_dim=int(cfg.get("linear_key_head_dim") or 0),
        linear_value_head_dim=int(cfg.get("linear_value_head_dim") or 0),
        linear_conv_kernel_dim=int(cfg.get("linear_conv_kernel_dim") or 4),
        rotary_dim=cfg.get("rotary_dim"),
        attn_output_gate=bool(cfg.get("attn_output_gate")),
        model_type=mt,
        sliding_window=int(cfg.get("sliding_window") or 0),
        global_head_dim=int(cfg.get("global_head_dim") or 0),
        num_global_key_value_heads=int(cfg.get("num_global_key_value_heads") or 0),
        rope_theta_swa=float(cfg.get("rope_theta_swa") or 0.0),
        partial_rotary_factor_full=float(cfg.get("partial_rotary_factor_full") or 1.0),
        final_logit_softcapping=float(cfg.get("final_logit_softcapping") or 0.0),
        hidden_activation=str(cfg.get("hidden_activation") or cfg.get("hidden_act") or "silu"),
        attention_k_eq_v=bool(cfg.get("attention_k_eq_v")),
        attention_multiplier=(float(cfg["attention_multiplier"])
                              if cfg.get("attention_multiplier") else None),
        embedding_multiplier=float(cfg.get("embedding_multiplier") or 1.0),
        residual_multiplier=float(cfg.get("residual_multiplier") or 1.0),
        logits_scaling=float(cfg.get("logits_scaling") or 1.0),
        kv_lora_rank=int(cfg.get("kv_lora_rank") or 0),
        qk_rope_head_dim=int(cfg.get("qk_rope_head_dim") or 0),
        qk_nope_head_dim=int(cfg.get("qk_nope_head_dim") or 0),
        v_head_dim=int(cfg.get("v_head_dim") or 0),
        router_scoring=str(cfg.get("router_scoring") or "softmax"),
        routed_scaling_factor=float(cfg.get("routed_scaling_factor") or 1.0),
        eos_token_id=_as_id_list(cfg.get("eos_token_id")),
        bos_token_id=(cfg.get("bos_token_id")
                      if isinstance(cfg.get("bos_token_id"), int) else None),
        raw=cfg,
    )
    return spec
