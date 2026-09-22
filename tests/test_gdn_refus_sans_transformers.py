"""Refus propre d'un modèle hybride Gated DeltaNet quand `transformers` (ou
son support GDN) manque — Océane, à sec, ordre relayé par Jérôme sur
`Qwen3.8-27B` refusé en pratique : `gdn.py:30` importe
`torch_chunk_gated_delta_rule` de `transformers`, jamais déclaré dans
`pyproject.toml` (seuls les extras `dev`/`nvml` existaient), et
`loader.py` construisait `GatedDeltaNet` sans jamais consulter
`gdn_available()` — l'échec arrivait donc loin de la vraie cause, avec un
`ImportError` brut plutôt qu'un message d'installation.

Le point de contrôle hybride (1 couche `full_attention`, 1 couche
`linear_attention`) est réel : `load_model_spec` → `auto_plan` →
`convert_checkpoint` → `load_model`, comme le montage établi de
`conftest.py` (`tiny_checkpoint`/`converted`), avec un second type de
couche. `model_type` reste vide à dessein : les seuls modèles renommés
par `_QWEN35_RENOMMAGE` (`convert.py`) sont `qwen3_5`/`qwen3_next` — un
`model_type` neutre laisse les noms de tenseurs `linear_attn.*` déjà
dans la convention acvram, sans renommage à reproduire ici.

`gdn_available()` est monkeypatché dans les deux sens (Jérôme : « import
monkeypatché pour simuler l'absence ») plutôt que de dépendre de l'état
réel du venv de test — qui n'a de toute façon pas `transformers` ici,
mais un test qui ne le dirait pas explicitement mentirait sur ce qu'il
vérifie si quelqu'un l'installait un jour dans ce venv.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

import acvram.engine.gdn as gdn_module
from acvram.engine.config import load_model_spec
from acvram.engine.loader import load_model
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.convert import ConversionOptions, convert_checkpoint

H, INTER, NH, NKV, V = 64, 128, 4, 2, 256
NKV_LIN, NVH_LIN, DK, DV, KCONV = 2, 2, 16, 16, 4


def _checkpoint_hybride(tmp_path_factory):
    d = tmp_path_factory.mktemp("hf-gdn")
    json.dump({
        "architectures": ["Qwen3NextForCausalLM"], "hidden_size": H,
        "intermediate_size": INTER, "num_hidden_layers": 2,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
        "layer_types": ["full_attention", "linear_attention"],
        "linear_num_key_heads": NKV_LIN, "linear_num_value_heads": NVH_LIN,
        "linear_key_head_dim": DK, "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": KCONV,
    }, open(d / "config.json", "w"))

    hd = H // NH
    key_dim, val_dim = NKV_LIN * DK, NVH_LIN * DV
    conv_dim = 2 * key_dim + val_dim
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}

    p0 = "model.layers.0."
    sd[p0 + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
    sd[p0 + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
    sd[p0 + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
    sd[p0 + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
    sd[p0 + "mlp.gate_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16) * .02
    sd[p0 + "mlp.up_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16) * .02
    sd[p0 + "mlp.down_proj.weight"] = torch.randn(H, INTER, dtype=torch.bfloat16) * .02
    sd[p0 + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p0 + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)

    p1 = "model.layers.1."
    sd[p1 + "linear_attn.qkv.weight"] = torch.randn(conv_dim, H, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.gate.weight"] = torch.randn(val_dim, H, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.alpha.weight"] = torch.randn(NVH_LIN, H, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.beta.weight"] = torch.randn(NVH_LIN, H, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.out.weight"] = torch.randn(H, val_dim, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.conv1d.weight"] = torch.randn(conv_dim, KCONV, dtype=torch.bfloat16) * .02
    sd[p1 + "linear_attn.dt_bias.weight"] = torch.zeros(NVH_LIN, dtype=torch.bfloat16)
    sd[p1 + "linear_attn.a_log.weight"] = torch.zeros(NVH_LIN, dtype=torch.bfloat16)
    sd[p1 + "linear_attn.norm.weight"] = torch.ones(DV, dtype=torch.bfloat16)
    sd[p1 + "mlp.gate_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16) * .02
    sd[p1 + "mlp.up_proj.weight"] = torch.randn(INTER, H, dtype=torch.bfloat16) * .02
    sd[p1 + "mlp.down_proj.weight"] = torch.randn(H, INTER, dtype=torch.bfloat16) * .02
    sd[p1 + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p1 + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)

    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="module")
def converti_gdn(tmp_path_factory, target_rig):
    src = _checkpoint_hybride(tmp_path_factory)
    spec = load_model_spec(src, "gdn-test")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=128, max_concurrent_seqs=1))
    out = str(tmp_path_factory.mktemp("acvram-gdn"))
    convert_checkpoint(src, plan, ConversionOptions(out_dir=out), spec=spec)
    return out


def test_refuse_proprement_quand_gdn_indisponible(converti_gdn, monkeypatch):
    monkeypatch.setattr(gdn_module, "gdn_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[gdn\]'"):
        load_model(converti_gdn, dtype=torch.bfloat16, max_model_len=128,
                   device_override="cpu")


def test_charge_quand_gdn_disponible(converti_gdn, monkeypatch):
    """Le même point de contrôle, `gdn_available()` forcé à True (sans que
    `transformers` soit réellement installé dans ce venv) : le chargement
    doit dépasser le refus — la construction de `GatedDeltaNet` elle-même
    n'a besoin de `transformers` qu'à l'inférence (`forward`), jamais à la
    construction. Prouve que le refus est bien conditionné par
    `gdn_available()`, pas par un état qu'on ne contrôle pas."""
    monkeypatch.setattr(gdn_module, "gdn_available", lambda: True)
    loaded = load_model(converti_gdn, dtype=torch.bfloat16, max_model_len=128,
                        device_override="cpu")
    assert loaded.model is not None
