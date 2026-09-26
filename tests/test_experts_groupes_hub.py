"""verdict-reconversion-30b-bloquee-21-09 : la disposition hub transformers ≥ 5
des experts MoE (`experts.gate_up_proj` [E, H, 2I], `experts.down_proj`
[E, I, H], sans index) doit ressortir de `_adapt_hf` sous la forme que le
convertisseur écrit et contrôle : `experts.{e}.{gate,up,down}_proj.weight`.
Doit casser si la scission disparaît (les noms 3D passent tels quels et le
contrôle final `attendus` les refuse) ou si l'ordre gate/up ou la transposition
change (valeurs)."""
import torch

from acvram.engine.config import ModelSpec
from acvram.quant.convert import _adapt_hf, _scinder_experts_groupes

E, H, I = 3, 8, 4


def _spec():
    return ModelSpec(
        name="qvl-moe-test", model_type="qwen3_vl_moe", architecture="qwen3_vl_moe",
        hidden_size=H, intermediate_size=I, num_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, vocab_size=16, max_position_embeddings=32,
        rope_theta=10000.0, moe_intermediate_size=I,
    )


def _source():
    torch.manual_seed(0)
    return [
        ("model.language_model.layers.0.self_attn.q_proj.weight", torch.randn(H, H)),
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", torch.randn(E, H, 2 * I)),
        ("model.language_model.layers.0.mlp.experts.down_proj", torch.randn(E, I, H)),
        ("model.language_model.layers.0.mlp.gate.weight", torch.randn(E, H)),
    ]


def test_scission_noms_et_valeurs():
    src = _source()
    gu, dn = src[1][1], src[2][1]
    sortie = dict(_adapt_hf(iter(src), _spec()))
    assert "model.layers.0.mlp.experts.gate_up_proj" not in sortie
    assert "model.layers.0.mlp.experts.down_proj" not in sortie
    for e in range(E):
        g = sortie[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        u = sortie[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
        d = sortie[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]
        assert g.shape == (I, H) and u.shape == (I, H) and d.shape == (H, I)
        assert torch.equal(g, gu[e, :, :I].t()) and torch.equal(u, gu[e, :, I:].t())
        assert torch.equal(d, dn[e].t())
        assert g.is_contiguous() and d.is_contiguous()
    # ce qui n'est pas un blob d'experts passe intact (nom et valeur)
    assert torch.equal(sortie["model.layers.0.mlp.gate.weight"], src[3][1])
    assert sortie["model.layers.0.self_attn.q_proj.weight"].shape == (H, H)


def test_contrat_du_controle_final_tenu():
    """Le contrôle final de `convert` attend `experts.0.gate_proj.weight` : la
    sortie de l'adaptateur doit le satisfaire (sans scission, il rend faux)."""
    noms = {n for n, _ in _adapt_hf(iter(_source()), _spec())}
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in noms
    assert "model.layers.0.mlp.experts.0.up_proj.weight" in noms


def test_experts_deja_indexes_intacts():
    """Un point de contrôle par expert (AWQ compressed-tensors, GGUF gemma4
    `experts.{e}.gate_up_proj` indexé) ne traverse pas la scission."""
    src = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(I, H)),
        ("model.layers.0.mlp.experts.0.gate_up_proj.weight", torch.randn(2 * I, H)),
        ("model.layers.0.mlp.experts.down_proj", torch.randn(I, H)),   # 2D : pas un blob hub
    ]
    assert [n for n, _ in _scinder_experts_groupes(iter(src))] == [n for n, _ in src]
