"""Assemble un modèle exécutable à partir des fragments acvram et d'un plan de
placement.

Le manifeste écrit par le convertisseur consigne, pour chaque tenseur, son
format et les clés qui le portent. Le chargement est donc mécanique : lire les
clés, reconstruire le conteneur quantifié, et le poser là où le plan l'indique —
résident sur un GPU, ou épinglé en mémoire hôte derrière un
:class:`~acvram.engine.layers.StreamedWeight`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, KVCacheConfig, PagedKVCache
from ..memory.tiering import Plan
from ..quant.calibrate import ChannelScaler
from ..quant.formats import INT8Tensor, PlainTensor
from ..quant.int4 import INT4Tensor
from ..quant.nvfp4 import NVFP4Tensor
from .config import ModelSpec
from .layers import QuantLinear, RMSNorm, RotaryEmbedding
from .model import (ACVRamModel, Attention, DecoderLayer, DecoderLayerGDN,
                    MLP, MoEBlock)

__all__ = ["LoadedModel", "load_model"]


class _ShardReader:
    """Accès paresseux aux tenseurs d'un ensemble de fragments safetensors."""

    def __init__(self, path: str, weight_map: dict[str, str]) -> None:
        self.path = path
        self.weight_map = weight_map
        self._open: dict[str, Any] = {}

    def get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        fn = self.weight_map.get(key)
        if fn is None:
            raise KeyError(f"{key} est absent de la table du manifeste")
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.path, fn),
                                       framework="pt", device="cpu")
        return self._open[fn].get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def close(self) -> None:
        self._open.clear()


class LoadedModel:
    def __init__(self, model: ACVRamModel, spec: ModelSpec, plan: Plan,
                 manifest: dict, path: str) -> None:
        self.model = model
        self.spec = spec
        self.plan = plan
        self.manifest = manifest
        self.path = path


def _build_quant(entry: dict, name: str, reader: _ShardReader,
                 group_size: int) -> Any:
    fmt = entry["format"]
    shape = tuple(entry["shape"])
    sd = {k.rsplit(".", 1)[-1]: reader.get(k) for k in entry["keys"]}
    if fmt == "nvfp4":
        return NVFP4Tensor(
            sd["qweight"], sd["block_scale"].view(torch.float8_e4m3fn),
            sd["global_scale"], shape, sd["qweight"].shape[-1] * 2)
    if fmt == "int4_awq":
        return INT4Tensor(sd["qweight"], sd["scales"], sd["zeros"],
                          entry.get("group_size", group_size), shape,
                          sd["qweight"].shape[-1] * 2)
    if fmt == "int8":
        return INT8Tensor(sd["qweight"], sd["scales"], sd["zeros"],
                          entry.get("group_size", group_size), shape)
    if fmt in ("bf16", "fp16"):
        return PlainTensor(sd["weight"], shape, fmt)
    raise KeyError(f"unknown format {fmt!r} for {name}")


def _build_scaler(entry: dict, name: str, reader: _ShardReader) -> Optional[ChannelScaler]:
    block = entry.get("hadamard_block", 0)
    scale = None
    key = f"{name}.act_scale"
    if entry.get("has_act_scale") and reader.has(key):
        scale = reader.get(key)
    if scale is None and not block:
        return None
    return ChannelScaler(scale, block)


def _linear(name: str, manifest: dict, reader: _ShardReader,
            group_size: int) -> Optional[QuantLinear]:
    entry = manifest["tensors"].get(name)
    if entry is None:
        return None
    q = _build_quant(entry, name, reader, group_size)
    scaler = _build_scaler(entry, name, reader)
    bias_key = name.replace(".weight", ".bias")
    bias = reader.get(bias_key) if reader.has(bias_key) else None
    return QuantLinear(q, bias, scaler, entry["shape"][0], entry["shape"][1])


def load_model(path: str, plan: Optional[Plan] = None,
               dtype: torch.dtype = torch.bfloat16,
               max_model_len: Optional[int] = None,
               device_override: Optional[str] = None) -> LoadedModel:
    """Charge en mémoire un répertoire de modèle converti, placé selon le plan."""
    with open(os.path.join(path, "acvram_manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    spec = ModelSpec(**{k: v for k, v in manifest["model"].items()
                        if k in ModelSpec.__dataclass_fields__})
    if plan is None:
        plan = _plan_from_manifest(manifest)
    reader = _ShardReader(path, manifest["weight_map"])
    group_size = manifest.get("options", {}).get("group_size", 128)

    def dev(name: str) -> torch.device:
        return torch.device(device_override or name)

    # plongements : une simple collecte, donc la RAM ne coûte qu'une petite copie par jeton
    embed = reader.get("model.embed_tokens.weight").to(dtype)
    embed_dev = dev(plan.embed_device) if plan.embed_device != "cpu" \
        else torch.device("cpu")
    embed = embed.to(embed_dev)

    rope = RotaryEmbedding(spec.rotary_dim or spec.head_dim,
                           spec.max_position_embeddings,
                           spec.rope_theta, spec.rope_scaling)

    layers: list[DecoderLayer] = []
    caches: dict[int, PagedKVCache] = {}
    kv_blocks = _kv_blocks_per_device(plan, spec, max_model_len)

    for lp in plan.layers:
        i = lp.index
        p = f"model.layers.{i}."
        d = dev(lp.exec_device)
        streamed_attn = lp.attn_storage == "cpu"
        streamed_mlp = lp.mlp_storage == "cpu"

        def lin(suffix: str, streamed: bool) -> QuantLinear:
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            return m.to_device(d, streamed=streamed)

        # Le MLP peut vivre et s'exécuter sur le processeur pendant que l'attention reste sur le GPU.
        mlp_on_cpu = (lp.mlp_storage == "cpu"
                      and getattr(lp, "mlp_exec", "gpu") == "cpu")
        mlp_dev = torch.device("cpu") if mlp_on_cpu else d
        streamed_mlp = streamed_mlp and not mlp_on_cpu

        def mlin(suffix: str) -> QuantLinear:
            m = _linear(p + suffix, manifest, reader, group_size)
            if m is None:
                raise KeyError(f"tenseur manquant {p + suffix}")
            return m.to_device(mlp_dev, streamed=streamed_mlp)

        def norm_opt(suffix: str) -> Optional[RMSNorm]:
            """RMSNorm facultative — absente des modeles sans QK-norm."""
            w = reader.get(p + suffix) if p + suffix in manifest["tensors"] else None
            return None if w is None else RMSNorm(w.to(dtype).to(d),
                                                  spec.rms_norm_eps)

        def faire_mlp() -> torch.nn.Module:
            if manifest["tensors"].get(p + "mlp.gate.weight") is None:
                return MLP(mlin("mlp.gate_proj.weight"),
                           mlin("mlp.up_proj.weight"),
                           mlin("mlp.down_proj.weight"))
            router = mlin("mlp.gate.weight")
            experts = []
            e = 0
            while manifest["tensors"].get(p + f"mlp.experts.{e}.gate_proj.weight"):
                experts.append(MLP(
                    mlin(f"mlp.experts.{e}.gate_proj.weight"),
                    mlin(f"mlp.experts.{e}.up_proj.weight"),
                    mlin(f"mlp.experts.{e}.down_proj.weight")))
                e += 1
            shared = None
            shared_gate = None
            if manifest["tensors"].get(p + "mlp.shared_expert.gate_proj.weight"):
                shared = MLP(
                    mlin("mlp.shared_expert.gate_proj.weight"),
                    mlin("mlp.shared_expert.up_proj.weight"),
                    mlin("mlp.shared_expert.down_proj.weight"))
                if manifest["tensors"].get(p + "mlp.shared_expert_gate.weight"):
                    shared_gate = reader.get(
                        p + "mlp.shared_expert_gate.weight").to(dtype).to(d)
            score_bias = None
            if manifest["tensors"].get(p + "mlp.gate.e_score_correction_bias"):
                score_bias = reader.get(
                    p + "mlp.gate.e_score_correction_bias").float().to(d)
            return MoEBlock(router, experts, spec.num_experts_per_tok or 2,
                            shared, shared_gate=shared_gate,
                            scoring=spec.router_scoring,
                            score_bias=score_bias,
                            routed_scale=spec.routed_scaling_factor)

        est_kimi = spec.model_type == "kimi_linear"
        if est_kimi:
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            petit16 = lambda suffix: reader.get(p + suffix).to(dtype).to(d)
            if spec.layer_types[i] == "linear_attention":
                from .kda import KimiDeltaAttention
                bloc = KimiDeltaAttention(
                    q_proj=lin("linear_attn.q_proj.weight", False),
                    k_proj=lin("linear_attn.k_proj.weight", False),
                    v_proj=lin("linear_attn.v_proj.weight", False),
                    out_proj=lin("linear_attn.out_proj.weight", False),
                    f_a=lin("linear_attn.f_a.weight", False),
                    f_b=lin("linear_attn.f_b.weight", False),
                    g_a=lin("linear_attn.g_a.weight", False),
                    g_b=lin("linear_attn.g_b.weight", False),
                    beta=lin("linear_attn.beta.weight", False),
                    conv_q=petit("linear_attn.conv1d_q.weight"),
                    conv_k=petit("linear_attn.conv1d_k.weight"),
                    conv_v=petit("linear_attn.conv1d_v.weight"),
                    dt_bias=petit("linear_attn.dt_bias.weight"),
                    a=petit("linear_attn.a.weight"),
                    norm_weight=petit("linear_attn.norm.weight"),
                    num_heads=spec.linear_num_value_heads,
                    head_dim=spec.linear_value_head_dim,
                    eps=spec.rms_norm_eps).to(d)
                bloc.fuse_projections()
            else:
                from .mla import MLAttention
                bloc = MLAttention(
                    q_proj=lin("self_attn.q_proj.weight", False),
                    kv_a_proj=lin("self_attn.kv_a_proj_with_mqa.weight", False),
                    o_proj=lin("self_attn.o_proj.weight", False),
                    kv_a_norm=petit16("self_attn.kv_a_layernorm.weight"),
                    k_b=petit16("self_attn.k_b_proj.weight"),
                    v_b=petit16("self_attn.v_b_proj.weight"),
                    num_heads=spec.num_attention_heads,
                    qk_nope=spec.qk_nope_head_dim,
                    qk_rope=spec.qk_rope_head_dim,
                    kv_lora_rank=spec.kv_lora_rank,
                    v_dim=spec.v_head_dim,
                    eps=spec.rms_norm_eps).to(d)
            mlp_kimi = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight"
                                         ).to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(
                reader.get(p + "post_attention_layernorm.weight"
                           ).to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, bloc, mlp_kimi,
                                          in_norm, post_norm, d))
            continue

        est_gdn = bool(spec.layer_types) and \
            spec.layer_types[i] == "linear_attention"
        if est_gdn:
            from .gdn import GatedDeltaNet
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            gdn = GatedDeltaNet(
                qkv=lin("linear_attn.qkv.weight", False),
                gate=lin("linear_attn.gate.weight", False),
                alpha=lin("linear_attn.alpha.weight", False),
                beta=lin("linear_attn.beta.weight", False),
                out=lin("linear_attn.out.weight", False),
                conv_weight=petit("linear_attn.conv1d.weight"),
                dt_bias=petit("linear_attn.dt_bias.weight"),
                # le convertisseur GGUF stocke -exp(A_log), pas A_log :
                # on inverse pour retrouver le paramètre de la référence
                a_log=torch.log(torch.clamp(
                    -petit("linear_attn.a_log.weight"), min=1e-12)),
                norm_weight=petit("linear_attn.norm.weight"),
                num_k_heads=spec.linear_num_key_heads,
                num_v_heads=spec.linear_num_value_heads,
                head_k_dim=spec.linear_key_head_dim,
                head_v_dim=spec.linear_value_head_dim,
                eps=spec.rms_norm_eps).to(d)
            mlp_gdn = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight"
                                         ).to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(
                reader.get(p + "post_attention_layernorm.weight"
                           ).to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, gdn, mlp_gdn, in_norm, post_norm, d))
            continue

        attn = Attention(
            spec,
            lin("self_attn.q_proj.weight", streamed_attn),
            lin("self_attn.k_proj.weight", streamed_attn),
            lin("self_attn.v_proj.weight", streamed_attn),
            lin("self_attn.o_proj.weight", streamed_attn),
            rope,
            norm_opt("self_attn.q_norm.weight"),
            norm_opt("self_attn.k_norm.weight"),
            output_gate=spec.attn_output_gate)

        mlp: torch.nn.Module = faire_mlp()

        in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d),
                          spec.rms_norm_eps)
        post_norm = RMSNorm(
            reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d),
            spec.rms_norm_eps)

        layers.append(DecoderLayer(i, attn, mlp, in_norm, post_norm, d, mlp_dev))

        n_blocks = kv_blocks.get(lp.exec_device, 0)
        if n_blocks:
            kv_fmt = next((t.kv_format for t in plan.tiers
                           if t.name == lp.exec_device), "int8")
            caches[i] = PagedKVCache(KVCacheConfig(
                num_layers=1, num_kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim, num_blocks=n_blocks,
                dtype=kv_fmt, device=str(d)))

    head_dev = dev(plan.lm_head_device) if plan.lm_head_device != "cpu" \
        else torch.device("cpu")
    norm = RMSNorm(reader.get("model.norm.weight").to(dtype).to(head_dev),
                   spec.rms_norm_eps)
    if manifest["tensors"].get("lm_head.weight"):
        lm_head = _linear("lm_head.weight", manifest, reader,
                          group_size).to_device(head_dev)
    else:
        # plongements partagés avec la sortie
        lm_head = QuantLinear(PlainTensor(embed.to(head_dev),
                                          tuple(embed.shape), "bf16"))
    reader.close()

    model = ACVRamModel(spec, embed, layers, norm, lm_head, caches, dtype)
    return LoadedModel(model, spec, plan, manifest, path)


def _kv_blocks_per_device(plan: Plan, spec: ModelSpec,
                          max_model_len: Optional[int]) -> dict[str, int]:
    """Répartit le budget KV de chaque appareil en blocs, partagés entre ses couches."""
    out: dict[str, int] = {}
    layers_on = {}
    for lp in plan.layers:
        layers_on[lp.exec_device] = layers_on.get(lp.exec_device, 0) + 1
    for dev, budget in plan.kv_budget.items():
        n_layers = max(1, layers_on.get(dev, 1))
        per_layer = budget // n_layers
        bytes_per_block = (2 * BLOCK_SIZE * spec.num_key_value_heads
                           * spec.head_dim + 2 * BLOCK_SIZE
                           * spec.num_key_value_heads * 2)
        out[dev] = max(1, per_layer // max(1, bytes_per_block))
    return out


def _plan_from_manifest(manifest: dict) -> Plan:
    from ..memory.tiering import LayerPlacement, Plan as _Plan, Tier
    d = manifest["plan"]
    plan = _Plan(model=d["model"])
    plan.tiers = [Tier(**t) for t in d["tiers"]]
    plan.layers = [LayerPlacement(**{k: v for k, v in l.items()
                                     if k not in ("streamed", "total_bytes",
                                                  "resident_bytes")})
                   for l in d["layers"]]
    plan.embed_device = d["embed_device"]
    plan.lm_head_device = d["lm_head_device"]
    plan.kv_budget = d.get("kv_budget", {})
    plan.kv_bytes_per_token = d.get("kv_bytes_per_token", 0)
    plan.kv_max_tokens = d.get("kv_max_tokens", 0)
    return plan
