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
    # nemotron_h : Mamba2 (SSD) + attention sans RoPE + MLP ReLU²
    mamba_num_heads: int = 0
    mamba_head_dim: int = 0
    mamba_n_groups: int = 1
    mamba_state_size: int = 128
    mamba_conv_kernel: int = 4
    attention_rope: bool = True
    # muse_glimmer : epsilon des normes post (1e-8)
    post_norm_eps: float = 0.0
    # starcoder2 : normes LayerNorm (biais), MLP non gaté GELU avec biais
    norm_type: str = "rms_norm"
    mlp_gated: bool = True
    # lfm2 : longueur du noyau de la conv courte
    conv_L_cache: int = 3
    # granite : multiplicateurs scalaires (attention, plongement, résidu, logits)
    attention_multiplier: Optional[float] = None
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    # kimi-linear : KDA + MLA + routeur DeepSeek
    model_type: str = ""
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    mla_rope: bool = False                # RoPE sur la partie pe (DeepSeek), pas Kimi
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
    "Lfm2ForCausalLM": "llama",
    "NemotronHForCausalLM": "llama",
    "FalconH1ForCausalLM": "llama",
    "Starcoder2ForCausalLM": "llama",
    "MuseGlimmerForConditionalGeneration": "llama",
    "Ernie4_5_MoeForCausalLM": "llama",
    "Lfm2MoeForCausalLM": "moe",
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
    rp_gen = cfg.get("rope_parameters") or {}
    if "rope_theta" not in cfg and isinstance(rp_gen, dict) and rp_gen.get("rope_theta"):
        cfg = {**cfg, "rope_theta": rp_gen["rope_theta"]}
    if cfg.get("model_type") == "gemma4_unified_text":
        cfg = {**cfg, "model_type": "gemma4_text"}     # même modèle texte
    if cfg.get("model_type") == "nemotron_h" and cfg.get("hybrid_override_pattern"):
        # point de contrôle HF (bf16 ou EXL3) : motif M/E/-/* → types de
        # couches, mêmes conventions que la synthèse GGUF (attention sans
        # RoPE, MLP ReLU², routage sigmoïde + biais, experts partagés)
        motif = str(cfg["hybrid_override_pattern"])
        kinds = {"M": "mamba", "E": "moe", "-": "mlp", "*": "full_attention"}
        cfg = {**cfg,
               "layer_types": [kinds[c] for c in motif[:int(cfg["num_hidden_layers"])]],
               "attention_rope": False, "hidden_act": "relu2",
               "rms_norm_eps": cfg.get("norm_eps") or cfg.get("layer_norm_epsilon") or 1e-5,
               "num_experts": cfg.get("n_routed_experts") or 0,
               "shared_expert_intermediate_size":
                   int(cfg.get("moe_shared_expert_intermediate_size") or 0) * int(cfg.get("n_shared_experts") or 0),
               "router_scoring": "sigmoid", "first_k_dense_replace": 0}
    if cfg.get("model_type") in ("lfm2", "lfm2_moe") and "norm_eps" in cfg:
        cfg = {**cfg, "rms_norm_eps": cfg["norm_eps"],
               "first_k_dense_replace": cfg.get("num_dense_layers") or 0,
               "router_scoring": "sigmoid"}
    if cfg.get("model_type") == "ernie4_5_moe":
        # MoE ERNIE : top-k sur softmax + biais de correction (sélection
        # seule), renormalisation, experts partagés fusionnés en un MLP,
        # première couche dense ; RoPE entrelacé → q/k dé-permutés à la
        # conversion (comme les GGUF llama)
        cfg = {**cfg, "num_experts": cfg.get("moe_num_experts"),
               "num_experts_per_tok": cfg.get("moe_k"),
               "shared_expert_intermediate_size":
                   int(cfg.get("moe_intermediate_size") or 0) * int(cfg.get("moe_num_shared_experts") or 0),
               "first_k_dense_replace": cfg.get("moe_layer_start_index") or 0,
               "scoring_func": "softmax", "norm_topk_prob": True}
    if cfg.get("model_type") in ("muse_glimmer", "muse_glimmer_text"):
        # attention à porte de sortie (gate_proj fusionné dans q_proj à la
        # conversion), q/k normalisés sans poids (facteur 3.87 replié dans
        # q_norm), logits × output_multiplier puis softcap
        cfg = {**cfg, "attn_output_gate": True, "model_type": "muse_glimmer",
               "hidden_activation": cfg.get("hidden_activation") or "silu",
               "logits_scaling": 1.0 / float(cfg.get("output_multiplier") or 1.0)}

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
    if mt in ("deepseek_v2", "deepseek_v3", "glm4_moe") and cfg.get("kv_lora_rank"):
        cfg = {**cfg, "mla_rope": True,
               "layer_types": cfg.get("layer_types") or
               ["full_attention"] * int(cfg.get("num_hidden_layers", 0))}
        if not cfg.get("shared_expert_intermediate_size") and cfg.get("n_shared_experts"):
            # point de contrôle HF : experts partagés fusionnés en un MLP
            cfg = {**cfg, "shared_expert_intermediate_size":
                   int(cfg.get("moe_intermediate_size") or 0) * int(cfg["n_shared_experts"]),
                   "router_scoring": cfg.get("router_scoring") or cfg.get("scoring_func") or "softmax"}
    if mt in ("qwen3_next", "kimi_linear", "qwen3_5", "qwen3_5_text",
              "qwen3_5_moe", "qwen3_5_moe_text") \
            and cfg.get("linear_num_value_heads"):
        if cfg.get("rotary_dim") is None and cfg.get("partial_rotary_factor"):
            hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
            cfg = {**cfg, "rotary_dim": int(hd * float(cfg["partial_rotary_factor"]))}
    elif (mt in ("kimi_linear", "mamba", "mamba2", "jamba")
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
        rms_norm_eps=cfg.get("rms_norm_eps", cfg.get("norm_epsilon", 1e-5)),
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
        mamba_num_heads=int(cfg.get("mamba_num_heads") or 0),
        mamba_head_dim=int(cfg.get("mamba_head_dim") or 0),
        mamba_n_groups=int(cfg.get("n_groups") or cfg.get("mamba_n_groups") or 1),
        mamba_state_size=int(cfg.get("ssm_state_size") or cfg.get("mamba_state_size") or 128),
        mamba_conv_kernel=int(cfg.get("conv_kernel") or cfg.get("mamba_conv_kernel") or 4),
        attention_rope=bool(cfg.get("attention_rope", True)),
        post_norm_eps=float(cfg.get("post_norm_eps") or 0.0),
        norm_type=str(cfg.get("norm_type") or "rms_norm"),
        mlp_gated=bool(cfg.get("mlp_gated", cfg.get("model_type") != "starcoder2")),
        conv_L_cache=int(cfg.get("conv_L_cache") or 3),
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
        q_lora_rank=int(cfg.get("q_lora_rank") or 0),
        mla_rope=bool(cfg.get("mla_rope")),
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
    _affiner_couches(spec, path)
    return spec


def _formes_du_point_de_controle(path: str) -> dict:
    """Nom et forme de chaque tenseur, sans charger un seul poids.

    Trois sources, par ordre de préférence : le manifeste d'un modèle déjà
    converti, l'en-tête d'un fichier GGUF, les en-têtes des fragments
    safetensors d'un dépôt Hugging Face. Tous trois portent les formes en
    clair, à quelques kilooctets de lecture.
    """
    if path.endswith(".json"):
        return {}
    manifeste = os.path.join(path, "acvram_manifest.json")
    if os.path.isfile(manifeste):
        try:
            with open(manifeste, "r", encoding="utf-8") as fh:
                t = json.load(fh)["tensors"]
            return {n: e["shape"] for n, e in t.items()
                    if isinstance(e, dict) and e.get("shape")}
        except Exception:                              # noqa: BLE001
            return {}
    if os.path.isfile(path) or not os.path.isdir(path):
        try:
            from ..quant.gguf import GGUFFile, is_gguf
            if not is_gguf(path):
                return {}
            g = GGUFFile(path)
        except Exception:                              # noqa: BLE001
            return {}
        # Les noms de llama.cpp (« blk.3.ffn_up_exps.weight ») ne ressemblent pas
        # à ceux de Hugging Face, et les experts y sont empilés en un tenseur
        # [E, ...] par projection. On traduit ce qu'il faut pour peser une
        # couche : sa famille et le nombre de paramètres qu'elle porte.
        formes: dict = {}
        for nom, info in g.tensors.items():
            dims = list(info[0])
            if not nom.startswith("blk."):
                continue
            morceaux = nom.split(".", 2)
            if len(morceaux) < 3:
                continue
            idx, reste = morceaux[1], morceaux[2]
            if "_exps" in reste:
                # [E, sortie, entrée] : une entrée par expert, pour que le
                # compte des experts et leur taille soient tous deux justes
                e = dims[0] if len(dims) == 3 else 1
                par_expert = dims[1:] if len(dims) == 3 else dims
                proj = reste.split(".")[0].replace("ffn_", "").replace("_exps", "")
                for k in range(int(e)):
                    formes[f"model.layers.{idx}.mlp.experts.{k}.{proj}_proj.weight"] = par_expert
            elif reste.startswith("ffn_"):
                formes[f"model.layers.{idx}.mlp.{reste}"] = dims
            elif "norm" in reste:
                formes[f"model.layers.{idx}.{reste}"] = dims
            else:
                formes[f"model.layers.{idx}.self_attn.{reste}"] = dims
        return formes
    brut: dict = {}
    try:
        import struct
        fragments = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        for f in fragments:
            with open(os.path.join(path, f), "rb") as fh:
                taille = struct.unpack("<Q", fh.read(8))[0]
                if taille > 200 * 1024 * 1024:         # en-tête aberrant
                    return {}
                entete = json.loads(fh.read(taille))
            for n, e in entete.items():
                if n != "__metadata__" and isinstance(e, dict) and e.get("shape"):
                    brut[n] = e["shape"]
    except Exception:                                  # noqa: BLE001
        return {}
    return _traduire_exl3(brut) if any(n.endswith(".suh") for n in brut) else brut


def _traduire_exl3(brut: dict) -> dict:
    """Rend les formes logiques d'un dépôt exllamav3.

    EXL3 ne range pas des matrices mais des treillis : ``…suh`` porte la
    dimension d'entrée, ``…svh`` la sortie, ``…trellis`` les poids compressés.
    Le produit des deux premières donne le nombre de paramètres de la
    projection. Les noms suivent aussi une autre convention — ``backbone``
    pour ``model``, ``mixer`` pour l'attention ou le bloc à experts.
    """
    formes: dict = {}
    for nom, forme in brut.items():
        if nom.endswith(".suh"):
            base = nom[:-4]
            sortie = brut.get(base + ".svh")
            if not sortie:
                continue
            logique = base.replace("backbone.", "model.")
            if ".mixer.experts." in logique or ".mixer.shared_experts" in logique:
                logique = logique.replace(".mixer.", ".mlp.")
            elif ".mixer." in logique:
                logique = logique.replace(".mixer.", ".self_attn.")
            formes[logique + ".weight"] = [int(sortie[0]), int(forme[0])]
        elif len(forme) == 2 and nom.endswith(".weight"):
            formes[nom.replace("backbone.", "model.")] = forme
    return formes


def _affiner_couches(spec: ModelSpec, path: str) -> None:
    """Recompte les paramètres par couche depuis les tenseurs réels du modèle.

    ``_build_layer`` déduit la taille d'une couche de la configuration, en
    supposant que toutes se ressemblent : une attention plus un MLP, MoE au-delà
    de ``first_k_dense_replace``. Les architectures hybrides démentent cette
    supposition — Nemotron-H alterne 23 couches Mamba sans MLP, 23 couches MoE
    sans attention et 6 couches d'attention pure, et le compte analytique
    donnait 103 milliards de paramètres pour un modèle qui en a 31,6. Le
    planificateur croyait alors devoir exiler 27 Gio en mémoire vive, étalait
    le modèle sur les deux cartes, et le décodage tombait à 26 jetons/s contre
    170 pour llama.cpp (mesuré le 5 septembre 2026).

    Quand le répertoire est un modèle déjà converti, son manifeste donne la
    forme exacte de chaque tenseur : on s'en sert. C'est juste pour toute
    architecture, présente ou à venir, sans rien deviner.
    """
    tenseurs = _formes_du_point_de_controle(path)
    if not tenseurs:
        return
    attn: dict[int, int] = {}
    mlp: dict[int, int] = {}
    norm: dict[int, int] = {}
    partage: dict[int, int] = {}
    experts: dict[int, set] = {}
    for nom, forme in tenseurs.items():
        if not nom.startswith("model.layers."):
            continue
        n = 1
        for x in forme:
            n *= int(x)
        try:
            i = int(nom.split(".")[2])
        except ValueError:
            continue
        if "norm" in nom.rsplit(".", 2)[-2:][0] or nom.endswith("norm.weight"):
            norm[i] = norm.get(i, 0) + n
        elif ".mlp." in nom or ".feed_forward." in nom:
            mlp[i] = mlp.get(i, 0) + n
            if ".experts." in nom:
                experts.setdefault(i, set()).add(nom.split(".experts.")[1].split(".")[0])
            else:
                partage[i] = partage.get(i, 0) + n      # routeur et expert partagé
        else:
            attn[i] = attn.get(i, 0) + n
    if not attn and not mlp:
        return
    couches = []
    for l in spec.layers:
        i = l.index
        if i not in attn and i not in mlp:
            couches.append(l)                           # couche absente : on garde l'estimation
            continue
        n_ex = len(experts.get(i, ()))
        couches.append(LayerSpec(i, attn.get(i, 0), mlp.get(i, 0), norm.get(i, l.norm_params),
                                 n_ex > 0, n_ex,
                                 min(l.n_experts_active, n_ex) if n_ex else 0,
                                 partage.get(i, 0)))
    spec.layers = couches
