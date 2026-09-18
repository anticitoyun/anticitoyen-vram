"""P1 — GEMM groupée classe Marlin au préfill (`ACVRAM_PREFILL_GROUPED=marlin`,
port vLLM v0.29.0). À sec : le Plan compte la seconde disposition ; le
régime la nomme (`experts_layout`). Sur carte : la sortie de
`MoEBlock._forward_prefill_grouped` sous « marlin » égale celle de « groupe »
(B0) à 2⁻⁷ × Σ|x·w| ; bras cassant : échelle de bloc décalée d'un rang → rouge."""
import os

import pytest
import torch

from acvram.engine import loader as LD
from acvram.engine.config import ModelSpec
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _spec():
    return ModelSpec(name="c30", architecture="Qwen3MoeForCausalLM", hidden_size=2048, intermediate_size=6144,
                     num_layers=48, num_attention_heads=32, num_key_value_heads=4, vocab_size=151936,
                     max_position_embeddings=32768, head_dim=128, num_experts=128, num_experts_per_tok=8,
                     moe_intermediate_size=768)


def _plan():
    tier = Tier(name="gpu-test", kind="gpu", device_index=0, capacity=30 * GIB,
                weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)
    couches = [LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                              fmt="nvfp4", attn_bytes=int(0.02 * GIB), mlp_bytes=int(0.3 * GIB),
                              mlp_active_bytes=0, is_moe=True) for i in range(48)]
    return Plan(model="synthetique", tiers=[tier], layers=couches)


def test_le_plan_compte_la_seconde_disposition_sous_marlin(monkeypatch):
    spec, plan = _spec(), _plan()
    monkeypatch.delenv("ACVRAM_PREFILL_GROUPED", raising=False)
    simple = LD._reserve_prefill(spec, 2048, {}, plan)
    monkeypatch.setenv("ACVRAM_PREFILL_GROUPED", "marlin")
    double = LD._reserve_prefill(spec, 2048, {}, plan)
    assert double - simple == 48 * int(0.3 * GIB)               # une copie des experts (Coder : ≈ 1,7 Gio... ici 14,4 Gio synthétiques)


def test_le_regime_nomme_la_disposition(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    engine = Engine(load_model(converted, dtype=torch.float32, device_override="cpu"), None,
                    max_batch_size=2, max_model_len=256)
    assert "experts_layout=simple" in engine.regime_ligne()


CARTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise (noyaux Marlin)")


def _bloc_moe_jouet(E=8, H=256, I=128, top_k=2, dev="cuda"):
    """Un MoEBlock à experts NVFP4 aléatoires, construit comme dans
    test_moe_hadamard_pile (QuantLinear → to_device, MLP, routeur)."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.quant.nvfp4 import quantize_nvfp4
    from acvram.quant.formats import _quantize_int8

    def lin(o, i, graine):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(o, i, generator=g) * 0.05).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i).to_device(dev)
    experts = [MLP(lin(I, H, 10 * e + 1), lin(I, H, 10 * e + 2), lin(H, I, 10 * e + 3)) for e in range(E)]
    gen = torch.Generator().manual_seed(5)
    routeur = QuantLinear(_quantize_int8((torch.randn(E, H, generator=gen) * 0.02).to(torch.bfloat16), 128),
                          out_features=E, in_features=H).to_device(dev)
    return MoEBlock(routeur, experts, top_k).to(dev)


@CARTE
def test_marlin_egale_groupe_au_prefill_et_le_bras_casse(monkeypatch):
    from acvram.engine import model as MD
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("extension Marlin non compilée à sec")
    E, H, I, top_k, T = 8, 256, 128, 2, 96
    bloc = _bloc_moe_jouet(E, H, I, top_k)
    x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
    logits = bloc.router(x).float()
    topw, topi = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    monkeypatch.setattr(MD, "_PREFILL_GROUPED", "groupe")
    assert bloc._try_build_stacks()
    y_groupe = bloc._forward_prefill_grouped(x, topw, topi.to(torch.int32))
    monkeypatch.setattr(MD, "_PREFILL_GROUPED", "marlin")
    bloc._stacks_marlin = bloc._construire_marlin(bloc._stacks, bloc._stacks_awq, bloc._stacks_awq.get("hadamard", {}))
    assert bloc._stacks_marlin is not None
    y_marlin = bloc._forward_prefill_grouped(x, topw, topi.to(torch.int32))
    borne = (y_groupe.float().abs().mean() + y_groupe.float().abs()).clamp_min(1e-3)
    hors = int(((y_marlin.float() - y_groupe.float()).abs() > 2 ** -7 * borne).sum())
    assert hors == 0, hors
    # bras cassant : les échelles de bloc de gate décalées d'un rang → sortie fausse
    w, sc, g, k, m = bloc._stacks_marlin["gate_proj"]
    bloc._stacks_marlin["gate_proj"] = (w, torch.roll(sc, 1, dims=2).contiguous(), g, k, m)
    y_faux = bloc._forward_prefill_grouped(x, topw, topi.to(torch.int32))
    assert int(((y_faux.float() - y_groupe.float()).abs() > 2 ** -7 * borne).sum()) > 0
