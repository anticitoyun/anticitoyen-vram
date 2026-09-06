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
import sys
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
from .model import (ACVRamModel, Attention, DecoderLayer, DecoderLayerGDN, MoEBlockGemma,
                    DecoderLayerGemma, DecoderLayerParallel, MLP, MLP2, MoEBlock)

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


_noyaux_signales = False


def _avertir_noyaux() -> None:
    """Dit franchement quand les noyaux CUDA manquent.

    Le repli sur les implémentations de référence divise le débit par un ordre
    de grandeur, et il ne se signalait que par un ``warnings.warn`` noyé dans
    la sortie du chargement. Une compilation qui échoue — un nvcc trop ancien,
    un en-tête absent — passait ainsi inaperçue pendant des semaines.
    """
    global _noyaux_signales
    if _noyaux_signales or not torch.cuda.is_available():
        return
    _noyaux_signales = True
    from ..kernels import build_info
    info = build_info()
    if info.get("available"):
        return
    raison = (info.get("error") or "raison inconnue").strip().splitlines()
    print("\n[acvram] ATTENTION : les noyaux CUDA ne sont PAS disponibles.",
          file=sys.stderr)
    print("[acvram] le moteur tourne sur les implementations de reference, "
          "environ dix fois plus lentes.", file=sys.stderr)
    print(f"[acvram] cause : {raison[0][:300]}", file=sys.stderr)
    print("[acvram] verifiez `python -m acvram doctor`.\n", file=sys.stderr)


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
        plan = _plan_from_manifest(manifest, spec)
    _avertir_noyaux()
    reader = _ShardReader(path, manifest["weight_map"])
    group_size = manifest.get("options", {}).get("group_size", 128)

    def dev(name: str) -> torch.device:
        return torch.device(device_override or name)

    # plongements : une simple collecte, donc la RAM ne coûte qu'une petite copie par jeton
    embed = reader.get("model.embed_tokens.weight").to(dtype)
    embed_dev = dev(plan.embed_device) if plan.embed_device != "cpu" \
        else torch.device("cpu")
    embed = embed.to(embed_dev)
    if embed_dev.type == "cpu":
        # Sans copie, la table reste un mmap du fichier safetensors : chaque
        # jeton nouveau touche une page non chargée — une lecture disque de
        # 100 ms au milieu du décodage. Résidente en RAM épinglée, elle se
        # collecte en microsecondes et se copie sans étape intermédiaire.
        embed = embed.contiguous().clone().pin_memory()

    rope_gemma = None
    rope = RotaryEmbedding(spec.rotary_dim or spec.head_dim,
                           spec.max_position_embeddings,
                           spec.rope_theta, spec.rope_scaling)

    layers: list[DecoderLayer] = []
    caches: dict[int, PagedKVCache] = {}
    _borner_kv_par_la_vram(plan, manifest, dev)
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
                        p + "mlp.shared_expert_gate.weight").to(dtype).to(mlp_dev)
            score_bias = None
            if manifest["tensors"].get(p + "mlp.gate.e_score_correction_bias"):
                score_bias = reader.get(
                    p + "mlp.gate.e_score_correction_bias").float().to(mlp_dev)
            return MoEBlock(router, experts, spec.num_experts_per_tok or 2,
                            shared, shared_gate=shared_gate,
                            norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
                            scoring=spec.router_scoring,
                            score_bias=score_bias,
                            routed_scale=spec.routed_scaling_factor)

        if spec.model_type in ("gemma4", "gemma4_text"):
            if rope_gemma is None:
                rope_gemma = (
                    RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
                                    spec.rope_theta_swa or 1e4, None, d, dtype),
                    RotaryEmbedding(spec.global_head_dim, spec.max_position_embeddings,
                                    spec.rope_theta, None, d, dtype,
                                    n_active=int(spec.partial_rotary_factor_full
                                                 * spec.global_head_dim / 2)))
            local = spec.layer_types[i] == "sliding_attention"
            hd = spec.head_dim if local else spec.global_head_dim
            nkv = spec.num_key_value_heads if local else spec.num_global_key_value_heads
            a_v = manifest["tensors"].get(p + "self_attn.v_proj.weight") is not None
            attn = Attention(
                spec,
                lin("self_attn.q_proj.weight", streamed_attn),
                lin("self_attn.k_proj.weight", streamed_attn),
                lin("self_attn.v_proj.weight", streamed_attn) if a_v else None,
                lin("self_attn.o_proj.weight", streamed_attn),
                rope_gemma[0] if local else rope_gemma[1],
                norm_opt("self_attn.q_norm.weight"),
                norm_opt("self_attn.k_norm.weight"),
                n_kv_heads=nkv, head_dim=hd, scale=1.0,
                v_norm_eps=spec.rms_norm_eps, k_eq_v=not a_v,
                window=spec.sliding_window if local else 0)
            mlp_g = MLP(mlin("mlp.gate_proj.weight"), mlin("mlp.up_proj.weight"),
                        mlin("mlp.down_proj.weight"), act=spec.hidden_activation)
            n4 = lambda suffix: RMSNorm(reader.get(p + suffix).to(dtype).to(d), spec.rms_norm_eps)
            out_scale = None
            if manifest["tensors"].get(p + "layer_scalar.weight") is not None:
                out_scale = reader.get(p + "layer_scalar.weight").to(torch.float32).to(d).reshape(-1)[0]
            moe = n1 = n2 = p2 = None
            if manifest["tensors"].get(p + "mlp.gate.weight") is not None:
                experts = []
                e = 0
                while manifest["tensors"].get(p + f"mlp.experts.{e}.gate_proj.weight"):
                    experts.append(MLP(mlin(f"mlp.experts.{e}.gate_proj.weight"),
                                       mlin(f"mlp.experts.{e}.up_proj.weight"),
                                       mlin(f"mlp.experts.{e}.down_proj.weight"),
                                       act=spec.hidden_activation))
                    e += 1
                moe = MoEBlockGemma(
                    mlin("mlp.gate.weight"), experts, spec.num_experts_per_tok or 8,
                    reader.get(p + "mlp.router_scale.weight").to(torch.float32).to(mlp_dev),
                    reader.get(p + "mlp.per_expert_scale.weight").to(torch.float32).to(mlp_dev),
                    spec.rms_norm_eps)
                n1 = n4("post_feedforward_layernorm_1.weight")
                n2 = n4("post_feedforward_layernorm_2.weight")
                p2 = n4("pre_feedforward_layernorm_2.weight")
            layers.append(DecoderLayerGemma(
                i, attn, mlp_g, n4("input_layernorm.weight"),
                n4("post_attention_layernorm.weight"),
                n4("pre_feedforward_layernorm.weight"),
                n4("post_feedforward_layernorm.weight"), out_scale, d,
                moe=moe, post_ffn_norm_1=n1, post_ffn_norm_2=n2, pre_ffn_norm_2=p2))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = next((t.kv_format for t in plan.tiers
                               if t.name == lp.exec_device), "int8")
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=nkv, head_dim=hd,
                    num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "muse_glimmer":
            local = spec.layer_types[i] == "sliding_attention"
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope,
                             norm_opt("self_attn.q_norm.weight"),
                             norm_opt("self_attn.k_norm.weight"),
                             window=spec.sliding_window if local else 0,
                             output_gate=True)
            mlp_g = MLP(mlin("mlp.gate_proj.weight"), mlin("mlp.up_proj.weight"),
                        mlin("mlp.down_proj.weight"), act=spec.hidden_activation)
            eps_post = spec.post_norm_eps or spec.rms_norm_eps
            n4 = lambda suffix, e: RMSNorm(reader.get(p + suffix).to(dtype).to(d), e)
            layers.append(DecoderLayerGemma(
                i, attn, mlp_g, n4("input_layernorm.weight", spec.rms_norm_eps),
                n4("post_attention_layernorm.weight", eps_post),
                n4("pre_feedforward_layernorm.weight", spec.rms_norm_eps),
                n4("post_feedforward_layernorm.weight", eps_post), None, d))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = next((t.kv_format for t in plan.tiers if t.name == lp.exec_device), "int8")
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "starcoder2":
            from .layers import LayerNorm
            ln = lambda suffix: LayerNorm(
                reader.get(p + suffix + ".weight").to(dtype).to(d),
                reader.get(p + suffix + ".bias").to(dtype).to(d) if reader.has(p + suffix + ".bias") else None,
                spec.rms_norm_eps)
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope,
                             window=spec.sliding_window)
            mlp_s = MLP2(mlin("mlp.up_proj.weight"), mlin("mlp.down_proj.weight"),
                         spec.hidden_activation)
            couche = DecoderLayer(i, attn, mlp_s, ln("input_layernorm"),
                                  ln("post_attention_layernorm"), d, mlp_dev)
            layers.append(couche)
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = next((t.kv_format for t in plan.tiers if t.name == lp.exec_device), "int8")
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "falcon_h1":
            from .mamba2 import Mamba2Mixer
            petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
            cb = petit("mamba.conv1d.bias") if manifest["tensors"].get(p + "mamba.conv1d.bias") else None
            mamba = Mamba2Mixer(
                in_proj=lin("mamba.in_proj.weight", False), out_proj=lin("mamba.out_proj.weight", False),
                conv_weight=petit("mamba.conv1d.weight"), conv_bias=cb,
                dt_bias=petit("mamba.dt_bias.weight"), A=petit("mamba.A.weight"),
                D=petit("mamba.D.weight"), norm_weight=petit("mamba.norm.weight"),
                num_heads=spec.mamba_num_heads, head_dim=spec.mamba_head_dim,
                n_groups=spec.mamba_n_groups, state_size=spec.mamba_state_size,
                eps=spec.rms_norm_eps).to(d)
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn), rope)
            mlp_f = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerParallel(i, attn, mamba, mlp_f, in_norm, post_norm, d))
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = next((t.kv_format for t in plan.tiers if t.name == lp.exec_device), "int8")
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type == "nemotron_h":
            kind = spec.layer_types[i]
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            if kind == "mamba":
                from .mamba2 import Mamba2Mixer
                petit = lambda suffix: reader.get(p + suffix).to(torch.float32).to(d)
                cb = petit("mamba.conv1d.bias") if manifest["tensors"].get(p + "mamba.conv1d.bias") else None
                bloc = Mamba2Mixer(
                    in_proj=lin("mamba.in_proj.weight", False),
                    out_proj=lin("mamba.out_proj.weight", False),
                    conv_weight=petit("mamba.conv1d.weight"), conv_bias=cb,
                    dt_bias=petit("mamba.dt_bias.weight"), A=petit("mamba.A.weight"),
                    D=petit("mamba.D.weight"), norm_weight=petit("mamba.norm.weight"),
                    num_heads=spec.mamba_num_heads, head_dim=spec.mamba_head_dim,
                    n_groups=spec.mamba_n_groups, state_size=spec.mamba_state_size,
                    eps=spec.rms_norm_eps).to(d)
                layers.append(DecoderLayerGDN(i, bloc, None, in_norm, None, d, mlp_device=mlp_dev))
                continue
            if kind in ("mlp", "moe"):
                if kind == "mlp":
                    mlp_n = MLP2(mlin("mlp.up_proj.weight"), mlin("mlp.down_proj.weight"), "relu2")
                else:
                    router = mlin("mlp.gate.weight"); experts = []; e = 0
                    while manifest["tensors"].get(p + f"mlp.experts.{e}.up_proj.weight"):
                        experts.append(MLP2(mlin(f"mlp.experts.{e}.up_proj.weight"),
                                            mlin(f"mlp.experts.{e}.down_proj.weight"), "relu2"))
                        e += 1
                    shared = None
                    if manifest["tensors"].get(p + "mlp.shared_expert.up_proj.weight"):
                        shared = MLP2(mlin("mlp.shared_expert.up_proj.weight"),
                                      mlin("mlp.shared_expert.down_proj.weight"), "relu2")
                    bias = None
                    if manifest["tensors"].get(p + "mlp.gate.e_score_correction_bias"):
                        # le routage vit avec les experts (RAM hôte si le plan
                        # les y a mis) : même appareil que les scores
                        bias = reader.get(p + "mlp.gate.e_score_correction_bias").float().to(mlp_dev)
                    mlp_n = MoEBlock(router, experts, spec.num_experts_per_tok or 2, shared,
                                     norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
                                     scoring=spec.router_scoring, score_bias=bias,
                                     routed_scale=spec.routed_scaling_factor)
                couche = DecoderLayer(i, None, mlp_n, in_norm, None, d, mlp_dev)
                layers.append(couche)
                continue
            # attention (sans RoPE), cache paginé
            attn = Attention(spec, lin("self_attn.q_proj.weight", streamed_attn),
                             lin("self_attn.k_proj.weight", streamed_attn),
                             lin("self_attn.v_proj.weight", streamed_attn),
                             lin("self_attn.o_proj.weight", streamed_attn),
                             None if not spec.attention_rope else rope)
            couche = DecoderLayer(i, attn, None, in_norm, None, d, d)
            layers.append(couche)
            n_blocks = kv_blocks.get(lp.exec_device, 0)
            if n_blocks:
                kv_fmt = next((t.kv_format for t in plan.tiers if t.name == lp.exec_device), "int8")
                caches[i] = PagedKVCache(KVCacheConfig(
                    num_layers=1, num_kv_heads=spec.num_key_value_heads,
                    head_dim=spec.head_dim, num_blocks=n_blocks, dtype=kv_fmt, device=str(d)))
            continue

        if spec.model_type in ("lfm2", "lfm2_moe") and spec.layer_types[i] == "conv":
            from .lfm2 import LFM2ShortConv
            bloc = LFM2ShortConv(
                in_proj=lin("conv.in_proj.weight", False),
                out_proj=lin("conv.out_proj.weight", False),
                conv_weight=reader.get(p + "conv.conv.weight").to(torch.float32).to(d),
                dim=spec.hidden_size).to(d)
            mlp_c = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(reader.get(p + "post_attention_layernorm.weight").to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, bloc, mlp_c, in_norm, post_norm, d, mlp_device=mlp_dev))
            continue

        est_kimi = spec.model_type in ("kimi_linear", "deepseek_v2", "deepseek_v3", "glm4_moe")
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
                q_lora = manifest["tensors"].get(p + "self_attn.q_a_proj.weight") is not None
                rope_mla = None
                if spec.mla_rope:
                    if "rope_mla_partage" not in dir():
                        rope_mla_partage = RotaryEmbedding(
                            spec.qk_rope_head_dim, spec.max_position_embeddings,
                            spec.rope_theta, spec.rope_scaling, d, dtype)
                    rope_mla = rope_mla_partage
                bloc = MLAttention(
                    q_proj=None if q_lora else lin("self_attn.q_proj.weight", False),
                    q_a_proj=lin("self_attn.q_a_proj.weight", False) if q_lora else None,
                    q_a_norm=petit16("self_attn.q_a_layernorm.weight") if q_lora else None,
                    q_b_proj=lin("self_attn.q_b_proj.weight", False) if q_lora else None,
                    rope=rope_mla,
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
                rs = spec.rope_scaling or {}
                if str(rs.get("rope_type") or rs.get("type") or "") == "yarn":
                    # YaRN : la sortie d'attention est rescalée par mscale² (DeepSeek)
                    import math as _m
                    msc = float(rs.get("mscale_all_dim") or rs.get("mscale") or 0.0)
                    fac = float(rs.get("factor") or 1.0)
                    if msc and fac > 1.0:
                        bloc.scale = bloc.scale * (0.1 * msc * _m.log(fac) + 1.0) ** 2
                bloc.fuse_projections()
            mlp_kimi = faire_mlp()
            in_norm = RMSNorm(reader.get(p + "input_layernorm.weight"
                                         ).to(dtype).to(d), spec.rms_norm_eps)
            post_norm = RMSNorm(
                reader.get(p + "post_attention_layernorm.weight"
                           ).to(dtype).to(d), spec.rms_norm_eps)
            layers.append(DecoderLayerGDN(i, bloc, mlp_kimi,
                                          in_norm, post_norm, d, mlp_device=mlp_dev))
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
                # le convertisseur GGUF stocke -exp(A_log) (drapeau
                # gdn_a_log_negexp) ; les sources HF portent A_log tel quel
                a_log=(torch.log(torch.clamp(
                    -petit("linear_attn.a_log.weight"), min=1e-12))
                       if spec.raw.get("gdn_a_log_negexp")
                       else petit("linear_attn.a_log.weight")),
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
            layers.append(DecoderLayerGDN(i, gdn, mlp_gdn, in_norm, post_norm, d, mlp_device=mlp_dev))
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

        couche = DecoderLayer(i, attn, mlp, in_norm, post_norm, d, mlp_dev)
        couche.residual_multiplier = spec.residual_multiplier
        layers.append(couche)

        n_blocks = kv_blocks.get(lp.exec_device, 0)
        if n_blocks:
            kv_fmt = next((t.kv_format for t in plan.tiers
                           if t.name == lp.exec_device), "int8")
            caches[i] = PagedKVCache(KVCacheConfig(
                num_layers=1, num_kv_heads=spec.num_key_value_heads,
                head_dim=spec.head_dim, num_blocks=n_blocks,
                dtype=kv_fmt, device=str(d)))

    # Projections empilées : gate/up des MLP (denses, experts partagés) et
    # q/k/v de l'attention — une GEMV au lieu de deux ou trois par couche.
    # Les experts d'un MoE en sont exclus : ils passent par le chemin groupé,
    # qui empile déjà les 128 experts, et les fusionner un à un doublerait
    # leurs poids sans rien accélérer.
    for layer in layers:
        experts = {id(e) for m in layer.modules() if isinstance(m, MoEBlock)
                   for e in m.experts}
        for m in layer.modules():
            if isinstance(m, MLP) and id(m) not in experts:
                m.fuse()
            elif isinstance(m, Attention):
                m.fuse()

    head_dev = dev(plan.lm_head_device) if plan.lm_head_device != "cpu" \
        else torch.device("cpu")
    if spec.model_type == "starcoder2":
        from .layers import LayerNorm
        norm = LayerNorm(reader.get("model.norm.weight").to(dtype).to(head_dev),
                         reader.get("model.norm.bias").to(dtype).to(head_dev)
                         if reader.has("model.norm.bias") else None, spec.rms_norm_eps)
    else:
        norm = RMSNorm(reader.get("model.norm.weight").to(dtype).to(head_dev),
                       spec.rms_norm_eps)
    if manifest["tensors"].get("lm_head.weight"):
        lm_head = _linear("lm_head.weight", manifest, reader,
                          group_size).to_device(head_dev)
    else:
        # Plongements partagés avec la sortie. La table sert deux fois et pas de
        # la même façon : à l'entrée c'est un *gather* d'une ligne, à la sortie
        # une projection qui relit la matrice entière **à chaque jeton**. Sur
        # Qwen3-4B ce sont 742 Mio, soit 24 % des octets lus par jeton pour une
        # seule couche — mesuré à 900 µs par passe sur la 3080 Ti, dix pour cent
        # du temps de décodage. La table reste en bf16 pour le gather ; la
        # projection en prend une copie quantifiée, qui coûte de la place mais
        # divise sa lecture par deux (int8) ou par trois et demi (nvfp4).
        lm_head = QuantLinear(_tete_liee(embed.to(head_dev)))
    tetes_mtp = _charger_mtp(manifest, reader, spec, plan, group_size, dtype,
                             head_dev, rope, kv_blocks)
    reader.close()

    model = ACVRamModel(spec, embed, layers, norm, lm_head, caches, dtype)
    if tetes_mtp:
        model.mtp = tetes_mtp[0]
        print(f"[acvram] tête de prédiction multi-jetons chargée "
              f"({len(tetes_mtp)} disponible(s))")
    return LoadedModel(model, spec, plan, manifest, path)


def _charger_mtp(manifest: dict, reader: "_ShardReader", spec: ModelSpec,
                 plan: Plan, group_size: int, dtype: torch.dtype,
                 device: torch.device, rope, kv_blocks: dict[str, int]) -> list:
    """Construit les têtes ``nextn`` que la conversion a conservées.

    Absentes de la plupart des modèles ; leur absence n'est pas une erreur. Une
    tête coûte un bloc de transformeur — sur un 27B de soixante-quatre couches,
    un soixante-quatrième du modèle — et sert de brouillon spéculatif.
    """
    from .mtp import MTPHead, cles_mtp

    indices = cles_mtp(manifest)
    if not indices or os.environ.get("ACVRAM_MTP") == "non":
        return []

    tetes = []
    for n in indices:
        p = f"model.mtp.{n}."

        def lin(suffix: str):
            m = _linear(p + suffix, manifest, reader, group_size)
            return None if m is None else m.to_device(device)

        def norme(suffix: str):
            cle = p + suffix
            if not reader.has(cle):
                return None
            return RMSNorm(reader.get(cle).to(dtype).to(device),
                           spec.rms_norm_eps)

        q, k, v, o = (lin("self_attn.q_proj.weight"), lin("self_attn.k_proj.weight"),
                      lin("self_attn.v_proj.weight"), lin("self_attn.o_proj.weight"))
        eh = lin("eh_proj.weight")
        in_norm, post_norm = norme("input_layernorm.weight"), norme("post_attention_layernorm.weight")
        enorm, hnorm = norme("enorm.weight"), norme("hnorm.weight")
        fin = norme("shared_head_norm.weight") or norme("norm.weight")
        gate, up, down = (lin("mlp.gate_proj.weight"), lin("mlp.up_proj.weight"),
                          lin("mlp.down_proj.weight"))
        if None in (q, k, v, o, eh, in_norm, post_norm, enorm, hnorm, fin,
                    gate, up, down):
            print(f"[acvram] tête MTP {n} incomplète, ignorée")
            continue

        attn = Attention(spec, q, k, v, o, rope,
                         q_norm=norme("self_attn.q_norm.weight"),
                         k_norm=norme("self_attn.k_norm.weight"),
                         output_gate=spec.attn_output_gate)
        mlp = MLP(gate, up, down, spec.hidden_activation)
        couche = DecoderLayer(spec.num_layers + n, attn, mlp, in_norm,
                              post_norm, device)
        n_blocks = max(64, kv_blocks.get(str(device), 512) // 16)
        cache = PagedKVCache(KVCacheConfig(
            num_layers=1, num_kv_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim, num_blocks=n_blocks,
            dtype="int8", device=str(device)))
        tetes.append(MTPHead(couche, enorm, hnorm, eh, fin, cache, device))
    return tetes


# Format de la projection de sortie quand elle partage la table des
# plongements : « bf16 » ne quantifie rien, « int8 » ou « nvfp4 » prennent une
# copie quantifiée pour la sortie seule.
_TETE_LIEE = os.environ.get("ACVRAM_TETE_LIEE", "int8").lower()


def _par_tranches(w: torch.Tensor, quant, fmt: str, lignes: int = 8192) -> Any:
    """Quantifie un très gros tenseur par paquets de lignes.

    Les quantifieurs passent par une copie float32 du tenseur entier : sur une
    table de plongements de 152 000 lignes, ce sont 1,45 Gio d'un coup, qui ne
    tiennent pas toujours à côté du modèle déjà chargé. Les lignes sont
    indépendantes — l'échelle est par ligne et par groupe — donc les traiter par
    paquets donne bit pour bit le même résultat pour un pic borné.
    """
    from ..quant.formats import INT8Tensor
    from ..quant.nvfp4 import NVFP4Tensor
    morceaux = [quant(w[i:i + lignes]) for i in range(0, w.shape[0], lignes)]
    if len(morceaux) == 1:
        return morceaux[0]
    if fmt == "int8":
        return INT8Tensor(
            torch.cat([m.qweight for m in morceaux]),
            torch.cat([m.scales for m in morceaux]),
            torch.cat([m.zeros for m in morceaux]),
            morceaux[0].group_size, tuple(w.shape))
    # NVFP4 : l'échelle globale est propre à chaque paquet, on garde la plus
    # grande et on ne peut pas recoller sans requantifier — on refuse plutôt
    # que de rendre un tenseur faux.
    raise RuntimeError("nvfp4 par tranches : échelles globales incompatibles")


def _tete_liee(embed: torch.Tensor) -> Any:
    """Le poids de la projection de sortie tirée d'une table partagée."""
    plein = PlainTensor(embed, tuple(embed.shape), "bf16")
    if _TETE_LIEE == "bf16" or not embed.is_cuda:
        return plein
    # La copie quantifiée s'ajoute à la table, elle ne la remplace pas : le
    # gather d'entrée a toujours besoin des poids en 16 bits. Sur un modèle qui
    # remplit déjà la carte, mieux vaut le débit qu'on a qu'un OOM au
    # chargement — on exige le double de la copie en mémoire libre.
    libre = torch.cuda.mem_get_info(embed.device)[0]
    besoin = embed.numel() * (1 if _TETE_LIEE == "int8" else 0.6)
    if libre < 2 * besoin:
        print(f"[acvram] tête liée laissée en bf16 : {libre / 2**20:.0f} Mio "
              f"libres, il en faudrait {2 * besoin / 2**20:.0f}",
              file=sys.stderr, flush=True)
        return plein
    try:
        if _TETE_LIEE == "nvfp4":
            from ..quant.nvfp4 import quantize_nvfp4
            return _par_tranches(embed, lambda t: quantize_nvfp4(t), "nvfp4")
        if _TETE_LIEE == "int8":
            from ..quant.formats import _quantize_int8
            return _par_tranches(embed, lambda t: _quantize_int8(t, 128), "int8")
    except Exception as exc:                      # noqa: BLE001
        # Quantifier la tête est un gain de débit, jamais une condition de
        # chargement : un échec se dit et se replie sur la table bf16.
        print(f"[acvram] tête liée laissée en bf16 ({type(exc).__name__}: {exc})",
              file=sys.stderr, flush=True)
    return plein


# Ce que la carte doit garder hors poids et hors KV : contexte CUDA, activations
# du prefill, graphes capturés, tampons de spéculation. Mesuré sur la 5090 : la
# capture des graphes échoue dès que moins de ~1 Gio reste libre.
_KV_MARGE_MIN = 1536 * 2**20
_KV_MARGE_PART = 0.05


def _borner_kv_par_la_vram(plan: Plan, manifest: dict, dev) -> None:
    """Borne le budget KV de chaque GPU par ce qu'il a réellement de libre.

    Le budget du manifeste vient du planificateur, qui raisonne sur des tailles
    nominales et une capacité de plaque signalétique. Sur la carte, ce sont les
    poids réels qui comptent, plus tout ce que le plan ne voit pas. Résultat
    observé avant ce garde-fou : 50 Mio libres après le chargement d'un
    35B-A3B, capture des graphes CUDA impossible, décodage dégradé. Ici, le
    budget est ramené à ``libre − poids réels − marge`` quand il le dépasse.
    """
    if not torch.cuda.is_available() or not plan.kv_budget:
        return
    attn, mlp, embed, head = _octets_reels(manifest)
    for t in plan.tiers:
        if t.kind != "gpu" or t.name not in plan.kv_budget:
            continue
        try:
            d = dev(t.name)
            libre, capacite = torch.cuda.mem_get_info(d)
        except Exception:                       # noqa: BLE001
            continue
        poids = (embed if plan.embed_device == t.name else 0) \
            + (head if plan.lm_head_device == t.name else 0)
        for l in plan.layers:
            poids += attn.get(l.index, l.attn_bytes) if l.attn_storage == t.name else 0
            poids += mlp.get(l.index, l.mlp_bytes) if l.mlp_storage == t.name else 0
        marge = max(_KV_MARGE_MIN, int(_KV_MARGE_PART * capacite))
        borne = libre - poids - marge
        budget = int(plan.kv_budget[t.name])
        if borne < budget:
            plan.kv_budget[t.name] = max(0, borne)
            print(f"[acvram] budget KV de {t.name} borné par la VRAM libre : "
                  f"{budget / 2**30:.2f} → {max(0, borne) / 2**30:.2f} Gio "
                  f"(libre {libre / 2**30:.1f}, poids {poids / 2**30:.1f}, "
                  f"marge {marge / 2**30:.1f})", file=sys.stderr)


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


def _octets_reels(manifest: dict) -> tuple[dict, dict, int, int]:
    """Octets réels par couche (attention, MLP) d'après les formats du
    manifeste : le plan a été calculé avec le format nominal (nvfp4), mais
    la conversion promeut en int8 les tenseurs sous le plancher de SNR — un
    70B y gagne une dizaine de Gio que le plan ignore, d'où des OOM au
    chargement."""
    attn: dict = {}; mlp: dict = {}; embed = 0; head = 0
    for nom, e in manifest["tensors"].items():
        if not isinstance(e, dict):
            continue
        shape = e.get("shape") or []
        n = 1
        for x in shape:
            n *= int(x)
        bpw = float(e.get("bpw") or (16.0 if e.get("format") in ("bf16", "fp16", None) else 4.5))
        octets = int(n * bpw / 8)
        if nom.startswith("model.layers."):
            i = int(nom.split(".")[2])
            if ".mlp." in nom:
                mlp[i] = mlp.get(i, 0) + octets
            else:
                attn[i] = attn.get(i, 0) + octets
        elif nom.startswith("model.embed_tokens"):
            embed += octets
        elif nom.startswith("lm_head"):
            head += octets
    return attn, mlp, embed, head


def _reajuster_plan(plan: Plan, manifest: dict) -> None:
    """Fait descendre en RAM hôte les MLP des dernières couches d'un GPU
    dont les poids réels dépassent la capacité de l'étage."""
    attn, mlp, embed, head = _octets_reels(manifest)
    for t in plan.tiers:
        if t.kind != "gpu":
            continue
        dev = t.name
        for l in plan.layers:
            # une couche décrite par le manifeste mais sans tenseur « .mlp. »
            # n'a réellement pas de MLP (bloc Mamba2/GDN pur) : garder sa
            # taille nominale gonflait le plan de dizaines de Gio et exilait
            # en RAM hôte des experts qui tenaient sur la carte
            decrite = l.index in attn
            l.attn_bytes = attn.get(l.index, l.attn_bytes)
            l.mlp_bytes = mlp.get(l.index, 0 if decrite else l.mlp_bytes)
        def utilise() -> int:
            u = int((plan.kv_budget or {}).get(dev, 0))
            u += embed if plan.embed_device == dev else 0
            u += head if plan.lm_head_device == dev else 0
            for l in plan.layers:
                u += l.attn_bytes if l.attn_storage == dev else 0
                u += l.mlp_bytes if l.mlp_storage == dev else 0
            return u
        # marge pour le contexte CUDA, les activations et les piles d'experts :
        # la capacité de l'étage est déjà nette des réserves du plan, mais un
        # 70B chargé à 99 % tombait encore en OOM à l'allocation du KV
        marge = max(2 * 2**30, int(0.07 * t.capacity))
        deplacees = 0
        while utilise() > t.capacity - marge:
            cand = [l for l in plan.layers if l.mlp_storage == dev]
            if not cand:
                break
            l = cand[-1]
            l.mlp_storage = "cpu"
            # Descendre un MLP en RAM dit où il est STOCKÉ, pas où il est
            # CALCULÉ. Forcer ici le calcul sur processeur défaisait la
            # décision du planificateur à chaque chargement : c'est ce chemin
            # qui a mis seize couches de Qwen3-Coder-Next sur le processeur,
            # à 10,3 jetons par seconde. Tant que le débit de calcul hôte
            # n'est pas mesuré (host_gemm_gb_s), les poids traversent le bus.
            if hasattr(l, "mlp_exec"):
                l.mlp_exec = ("cpu" if os.environ.get("ACVRAM_MLP_HOTE_CPU")
                              else "gpu")
            deplacees += 1
        if deplacees:
            print(f"[acvram] plan réajusté : {deplacees} MLP de plus en RAM hôte sur {dev} "
                  f"(poids réels {utilise() / 2**30:.1f} Gio pour {t.capacity / 2**30:.1f} Gio)",
                  flush=True)
            continue
        # Symétrique de la descente. Le plan est figé au moment de la
        # conversion, avec le format nominal et la machine d'alors ; les
        # poids réels sont souvent plus compacts (nvfp4 à 4,5 bpw là où le
        # plan comptait 6). Une couche laissée en RAM hôte y coûte un aller
        # PCIe par jeton : dès qu'elle tient sur la carte, elle y remonte.
        if os.environ.get("ACVRAM_PLAN_FIGE"):
            continue
        remontees = 0
        for l in plan.layers:
            if l.exec_device != dev:
                continue
            if l.attn_storage == "cpu" and utilise() + l.attn_bytes <= t.capacity - marge:
                l.attn_storage = dev
            if l.mlp_storage != "cpu":
                continue
            if utilise() + l.mlp_bytes > t.capacity - marge:
                continue
            l.mlp_storage = dev
            if hasattr(l, "mlp_exec"):
                l.mlp_exec = "gpu"
            remontees += 1
        if remontees:
            print(f"[acvram] plan réajusté : {remontees} MLP remontés en VRAM sur {dev} "
                  f"(poids réels {utilise() / 2**30:.1f} Gio pour {t.capacity / 2**30:.1f} Gio)",
                  flush=True)
    _rapatrier_sur_une_carte(plan, attn, mlp, embed, head)


def _rapatrier_sur_une_carte(plan: Plan, attn: dict, mlp: dict,
                             embed: int, head: int) -> None:
    """Ramène tout le modèle sur la première carte quand il y tient.

    Le plan est figé à la conversion, avec le compte de paramètres d'alors.
    Pour les architectures hybrides, ce compte a longtemps triplé la taille
    réelle (Nemotron-H : 103 milliards annoncés pour 31,6), et le planificateur
    étalait sur deux cartes un modèle qui tenait sur une. Chaque frontière
    franchie coûte un aller-retour d'état caché par jeton, et la carte
    d'appoint est trois fois plus lente : mesuré le 5 septembre 2026,
    26 jetons/s au lieu de 170.

    Les octets réels sont connus ici : si la première carte les porte, avec son
    budget KV et la marge, les couches de la seconde y reviennent. Échappement
    par ``ACVRAM_PLAN_FIGE``.
    """
    if os.environ.get("ACVRAM_PLAN_FIGE"):
        return
    gpus = [t for t in plan.tiers if t.kind == "gpu"]
    if len(gpus) < 2:
        return
    occupes = {l.exec_device for l in plan.layers} & {t.name for t in gpus}
    if len(occupes) < 2:
        return
    principal = gpus[0]
    if any(l.attn_storage == "cpu" or l.mlp_storage == "cpu" for l in plan.layers):
        return                          # déjà à l'étroit : ne pas empirer
    besoin = embed if plan.embed_device == principal.name else 0
    besoin += head
    for l in plan.layers:
        besoin += attn.get(l.index, l.attn_bytes) + mlp.get(l.index, l.mlp_bytes)
    besoin += sum(int(v) for v in (plan.kv_budget or {}).values())
    marge = max(2 * 2**30, int(0.07 * principal.capacity))
    if besoin > principal.capacity - marge:
        return
    for l in plan.layers:
        l.exec_device = principal.name
        l.attn_storage = principal.name
        l.mlp_storage = principal.name
        if hasattr(l, "mlp_exec"):
            l.mlp_exec = "gpu"
    plan.lm_head_device = principal.name
    if plan.embed_device in {t.name for t in gpus}:
        plan.embed_device = principal.name
    plan.kv_budget = {principal.name: sum(int(v) for v in (plan.kv_budget or {}).values())}
    plan.stage_ranges = {principal.name: (0, len(plan.layers) - 1)}
    print(f"[acvram] plan réajusté : tout le modèle rapatrié sur {principal.name} "
          f"({besoin / 2**30:.1f} Gio pour {principal.capacity / 2**30:.1f} Gio) — "
          f"une frontière de moins par jeton", flush=True)


def _replanifier(manifest: dict, spec: "ModelSpec") -> "Plan | None":
    """Rejoue le planificateur quand le plan figé au manifeste ne décrit plus
    cette machine.

    Le plan est écrit une fois pour toutes à la conversion, et `_reajuster_plan`
    ne sait que faire *descendre* des MLP en RAM hôte : aucune amélioration du
    planificateur n'atteint jamais un modèle déjà converti. Le 7 septembre 2026,
    Qwen3-Coder-Next tournait à 10,3 jetons par seconde sur un plan qui ignorait
    la seconde carte et calculait seize couches sur le processeur, alors que le
    planificateur du jour recrutait les deux cartes et n'en exilait que quatre.
    On ne rejoue que si les cartes ont changé : sinon le plan figé fait foi, et
    les mesures du parc restent comparables.
    """
    if os.environ.get("ACVRAM_PLAN_FIGE") or os.environ.get("ACVRAM_SANS_REPLAN"):
        return None
    d = manifest["plan"]
    figees = {t["name"] for t in d.get("tiers", []) if t.get("kind") == "gpu"}
    try:
        from ..hardware.detect import detect_rig
        from ..memory.tiering import (PlannerOptions, auto_plan,
                                      empreinte_planificateur)
        rig = detect_rig()
        presentes = {f"cuda:{g.index}" for g in rig.gpus}
        # Deux raisons de replanifier, et il faut les deux : le matériel a
        # changé, ou le planificateur a changé. Sans la seconde, la prochaine
        # correction du modèle de coût laisserait tous les manifestes périmés
        # en silence, et le défaut se redécouvrirait par un modèle vingt fois
        # trop lent et une soirée perdue.
        empreinte = d.get("empreinte_planificateur")
        a_jour = empreinte == empreinte_planificateur()
        if not presentes or (figees == presentes and a_jour):
            return None
        motif = ("les cartes ont changé" if figees != presentes
                 else "le planificateur a changé depuis la conversion")
        ctx = max(2048, int(d.get("kv_max_tokens") or 0) or 8192)
        neuf, _ = auto_plan(spec, rig, PlannerOptions(max_model_len=ctx))
    except Exception as e:                                   # pragma: no cover
        print(f"[acvram] replanification impossible ({e}) ; plan du manifeste "
              f"conservé", flush=True)
        return None
    print(f"[acvram] plan recalculé : {motif} "
          f"(manifeste {sorted(figees) or 'aucun GPU'}, machine {sorted(presentes)})",
          flush=True)
    return neuf


def _plan_from_manifest(manifest: dict, spec: "ModelSpec | None" = None) -> Plan:
    from ..memory.tiering import LayerPlacement, Plan as _Plan, Tier
    d = manifest["plan"]
    if spec is not None:
        neuf = _replanifier(manifest, spec)
        if neuf is not None:
            _reajuster_plan(neuf, manifest)
            return neuf
    plan = _Plan(model=d["model"])
    plan.tiers = [Tier(**t) for t in d["tiers"]]
    plan.layers = [LayerPlacement(**{k: v for k, v in l.items()
                                     if k not in ("streamed", "total_bytes",
                                                  "resident_bytes")})
                   for l in d["layers"]]
    plan.embed_device = d["embed_device"]
    plan.lm_head_device = d["lm_head_device"]
    _reajuster_plan(plan, manifest)
    plan.kv_budget = d.get("kv_budget", {})
    plan.kv_bytes_per_token = d.get("kv_bytes_per_token", 0)
    plan.kv_max_tokens = d.get("kv_max_tokens", 0)
    return plan
