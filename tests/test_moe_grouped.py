"""Le chemin MoE groupé rend ce que rend la boucle par expert.

Trois lancements par couche contre trois par expert actif : même mathématique,
et un test qui l'affirme sur les logits, chemin contre chemin, sur le même
modèle chargé.
"""

import json

import pytest
import torch

pytestmark = pytest.mark.gpu_requis

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from conftest import assert_logits_proches

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="le chemin groupe est CUDA")


@pytest.fixture(scope="session")
def tiny_moe(tmp_path_factory):
    from safetensors.torch import save_file

    torch.manual_seed(20260901)
    H, I, L, NH, NKV, V, E, IK = 128, 256, 2, 4, 2, 512, 4, 96
    d = tmp_path_factory.mktemp("hf_moe")
    json.dump({
        "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6, "rope_theta": 100000.0,
        "torch_dtype": "bfloat16", "model_type": "qwen3_moe",
        "num_experts": E, "num_experts_per_tok": 2,
        "moe_intermediate_size": IK,
    }, open(d / "config.json", "w"))
    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .02}
    for i in range(L):
        p = f"model.layers.{i}."
        for n, sh in [("self_attn.q_proj", (NH * hd, H)),
                      ("self_attn.k_proj", (NKV * hd, H)),
                      ("self_attn.v_proj", (NKV * hd, H)),
                      ("self_attn.o_proj", (H, NH * hd))]:
            sd[p + n + ".weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .02
        sd[p + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16) * .02
        for e in range(E):
            for n, sh in [("gate_proj", (IK, H)), ("up_proj", (IK, H)),
                          ("down_proj", (H, IK))]:
                sd[p + f"mlp.experts.{e}.{n}.weight"] = \
                    torch.randn(*sh, dtype=torch.bfloat16) * .02
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .02
    save_file(sd, str(d / "model.safetensors"))

    from acvram.engine.config import load_model_spec
    from acvram.hardware.profiles import load_profile
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    spec = load_model_spec(str(d), "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    out = str(tmp_path_factory.mktemp("acvram_moe"))
    convert_checkpoint(str(d), plan, ConversionOptions(out_dir=out), spec=spec)
    return out


def _moe_blocks(model):
    from acvram.engine.model import MoEBlock
    return [m for m in model.modules() if isinstance(m, MoEBlock)]


@needs_cuda
def test_grouped_equals_loop(tiny_moe):
    loaded = load_model(tiny_moe, dtype=torch.bfloat16, device_override="cuda:0")
    blocks = _moe_blocks(loaded.model)
    assert blocks, "la fixture n'a pas produit de MoE"
    x = (torch.randn(2, 128, device="cuda:0") * 0.3).to(torch.bfloat16)

    y_groupe = blocks[0](x)
    assert blocks[0]._stack_state == "oui", "pile refusee sur un cas homogene"
    for b in blocks:
        b._stack_state = "non"                 # forcer la boucle par expert
    y_boucle = blocks[0](x)
    err = (y_groupe.float() - y_boucle.float()).norm() / \
        y_boucle.float().norm().clamp(min=1e-9)
    assert err < 3e-2, f"chemins groupe et boucle divergent : {err:.4f}"


@needs_cuda
def test_moe_generation_still_sane(tiny_moe):
    loaded = load_model(tiny_moe, dtype=torch.bfloat16, device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=2, max_model_len=128)
    outs = [t for o in e.generate([3, 1, 4, 1, 5],
                                  SamplingParams(temperature=0.0, max_tokens=8))
            for t in o.token_ids]
    assert len(outs) == 8


@needs_cuda
def test_moe_graph_equals_eager(tiny_moe):
    """Le rejeu en graphe d'une couche MoE groupée rend les logits de l'eager."""
    loaded = load_model(tiny_moe, dtype=torch.bfloat16, device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=2, max_model_len=128,
               enable_cuda_graphs=True)
    assert e.graphs is not None and e.graphs.enabled, \
        "MoE empilé refusé par les graphes"
    e.add_request([3, 1, 4, 1, 5], SamplingParams(temperature=0.0, max_tokens=24))
    e.step()
    for _ in range(6):
        dec = e._decodables()
        for s in dec:
            assert e._grow(s)
        batch = e._build_batch(dec, prefill=False)
        eager = e.model(batch).float()
        graphe = e.graphs.run(batch)
        assert graphe is not None
        assert_logits_proches(eager, graphe.float(),
                              "graphe et eager divergent sur une couche MoE")
        e._emit(graphe, dec)

