import json
import os

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Un point de contrôle de 4 couches, de forme llama, assez petit pour être
    converti en une seconde.

    Avec une graine fixe. Sans elle, toute la suite est non déterministe, et les
    tests qui comparent des formats de quantification sur ce modèle sont assez
    proches du plancher de bruit pour qu'un tirage différent les fasse basculer
    — c'est exactement ainsi qu'une exécution verte dans un répertoire de travail
    est devenue rouge dans une extraction propre.
    """
    from safetensors.torch import save_file

    torch.manual_seed(20260830)
    H, I, L, NH, NKV, V = 256, 688, 4, 8, 2, 1024
    d = tmp_path_factory.mktemp("hf")
    json.dump({
        "architectures": ["LlamaForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))

    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * 0.02}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
        sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
        sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


@pytest.fixture(scope="session")
def target_rig():
    from acvram.hardware.profiles import load_profile
    return load_profile("rig-14900k-5090-3080ti")


@pytest.fixture(scope="session")
def converted(tiny_checkpoint, target_rig, tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram"))
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out), spec=spec)
    return out
