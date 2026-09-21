"""verdict-p3-3-30b-reconv-21-09 : sur Qwen3-VL-30B (experts groupés du hub
transformers ≥ 5, `experts.gate_up_proj` [E, H, 2I] / `experts.down_proj`
[E, I, H]), la collecte relevait 0 tenseur — 18 432/18 432 experts sans
statistique, tout le MoE en repli échelle identité. Témoin : le MÊME modèle
écrit par expert. Les statistiques doivent être identiques dans les deux
dispositions et aucun expert ne doit manquer. Doit casser si la collecte
redevient aveugle au blob (KeyError à e = 0 → couche sans stats) ou si l'ordre
gate/up ou la transposition diverge de `convert.py::_expert_depuis_blob`."""
import json
import os

import pytest
import torch

from acvram.engine.config import load_model_spec
from acvram.quant.collect import collect_activation_stats
from acvram.quant.convert import _adapt_hf

H, I, L, NH, NKV, V, E, IK = 32, 48, 2, 2, 1, 64, 3, 16


def _config(d):
    json.dump({
        "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 100000.0,
        "torch_dtype": "bfloat16", "model_type": "qwen3_moe",
        "num_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": IK,
    }, open(os.path.join(d, "config.json"), "w"))


def _tenseurs():
    torch.manual_seed(20260921)
    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .2}
    for i in range(L):
        p = f"model.layers.{i}."
        for n, sh in [("self_attn.q_proj", (NH * hd, H)), ("self_attn.k_proj", (NKV * hd, H)),
                      ("self_attn.v_proj", (NKV * hd, H)), ("self_attn.o_proj", (H, NH * hd))]:
            sd[p + n + ".weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .2
        sd[p + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16) * .2
        for e in range(E):
            for n, sh in [("gate_proj", (IK, H)), ("up_proj", (IK, H)), ("down_proj", (H, IK))]:
                sd[p + f"mlp.experts.{e}.{n}.weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .2
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .2
    return sd


def _en_hub(sd):
    """Disposition hub ≥ 5 : `model.language_model.`, un blob par projection
    (mêmes formes que l'en-tête officiel du 30B : gate_up [E, H, 2I], down [E, I, H])."""
    out = {}
    for i in range(L):
        p = f"model.layers.{i}.mlp.experts."
        out[p + "gate_up_proj"] = torch.stack([
            torch.cat([sd[p + f"{e}.gate_proj.weight"], sd[p + f"{e}.up_proj.weight"]], 0).t()
            for e in range(E)]).contiguous()
        out[p + "down_proj"] = torch.stack([sd[p + f"{e}.down_proj.weight"].t() for e in range(E)]).contiguous()
    for k, t in sd.items():
        if ".mlp.experts." not in k:
            out[k] = t
    return {f"model.language_model.{k[len('model.'):]}" if k.startswith("model.") else k: t
            for k, t in out.items()}


@pytest.fixture(scope="module")
def deux_dispositions(tmp_path_factory):
    from safetensors.torch import save_file
    sd = _tenseurs()
    a = str(tmp_path_factory.mktemp("moe_par_expert"))
    b = str(tmp_path_factory.mktemp("moe_hub"))
    for d, tenseurs in ((a, sd), (b, _en_hub(sd))):
        _config(d)
        save_file(tenseurs, os.path.join(d, "model.safetensors"))
    return a, b


def test_collecte_voit_les_experts_groupes(deux_dispositions):
    par_expert, hub = deux_dispositions
    spec = load_model_spec(par_expert, "tiny-moe")
    calib = [[3, 5, 7, 11, 13, 17, 19, 23], [2, 4, 6, 8, 10, 12, 14, 16]]
    ref = collect_activation_stats(par_expert, spec, calib, device="cpu", dtype=torch.float32)
    mm = collect_activation_stats(hub, spec, calib, device="cpu", dtype=torch.float32)

    attendus = {f"model.layers.{i}.mlp.experts.{e}.{n}.weight"
                for i in range(L) for e in range(E) for n in ("gate_proj", "up_proj", "down_proj")}
    assert attendus <= set(ref), "témoin : la disposition par expert doit déjà statistiquer chaque expert"
    assert set(mm) == set(ref), sorted(set(ref) ^ set(mm))[:5]
    for nom in attendus:
        assert mm[nom].n_samples > 0, f"{nom} : 0 échantillon (expert sans stats)"
        assert torch.equal(mm[nom].mean_abs, ref[nom].mean_abs), nom
        assert mm[nom].n_samples == ref[nom].n_samples


def test_collecte_et_flux_principal_lisent_le_meme_poids(deux_dispositions):
    """Le poids que la collecte statistique sous un nom est celui que le flux
    principal quantifie sous ce nom — sinon l'échelle AWQ va au mauvais tenseur."""
    from safetensors import safe_open
    par_expert, hub = deux_dispositions
    spec = load_model_spec(par_expert, "tiny-moe")
    with safe_open(os.path.join(hub, "model.safetensors"), framework="pt", device="cpu") as fh:
        flux = dict(_adapt_hf(iter([(k, fh.get_tensor(k)) for k in fh.keys()]), spec))
    with safe_open(os.path.join(par_expert, "model.safetensors"), framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if ".mlp.experts." in k:
                assert torch.equal(flux[k], fh.get_tensor(k)), k
