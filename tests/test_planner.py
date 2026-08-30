"""Les décisions du planificateur de placement, sur la vraie machine cible."""

import json
import os

import pytest

from acvram.engine.config import ModelSpec, load_model_spec
from acvram.hardware.detect import capabilities_for_sm
from acvram.memory.tiering import PlannerOptions, auto_plan, plan_placement


def _spec(tmp_path, name, **cfg):
    base = dict(architectures=["LlamaForCausalLM"], hidden_size=4096,
                intermediate_size=11008, num_hidden_layers=32,
                num_attention_heads=32, num_key_value_heads=8,
                vocab_size=128256, max_position_embeddings=8192)
    base.update(cfg)
    d = tmp_path / name
    d.mkdir()
    json.dump(base, open(d / "config.json", "w"))
    return load_model_spec(str(d), name)


def test_blackwell_a_le_fp4_ampere_non():
    assert capabilities_for_sm(120).weight_format == "nvfp4"
    assert capabilities_for_sm(120).fp4_tensor_core
    assert capabilities_for_sm(86).weight_format == "int4_awq"
    assert not capabilities_for_sm(86).fp4_tensor_core
    assert not capabilities_for_sm(86).fp8_tensor_core


def test_second_gpu_is_left_idle_when_the_model_fits_on_the_first(tmp_path, target_rig):
    """Étendre le pipeline à une carte plus lente coûte du débit mono-flux."""
    spec = _spec(tmp_path, "small", num_hidden_layers=24)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert {lp.exec_device for lp in plan.layers} == {"cuda:0"}
    assert any("laisse oisif a dessein" in w for w in plan.warnings)


def test_big_model_uses_both_gpus(tmp_path, target_rig):
    spec = _spec(tmp_path, "big", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert {lp.exec_device for lp in plan.layers} == {"cuda:0", "cuda:1"}


def test_pipeline_crosses_gpus_exactly_once(tmp_path, target_rig):
    spec = _spec(tmp_path, "cross", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    crossings = sum(1 for a, b in zip(plan.layers, plan.layers[1:])
                    if a.exec_device != b.exec_device)
    assert crossings <= 1


def test_attention_is_pinned_before_mlp_spills(tmp_path, target_rig):
    """L'attention est petite et critique en latence ; c'est le MLP qui part en RAM."""
    # 128 experts de 1536 sur 60 couches font environ 145 milliards de
    # paramètres : bien au-delà des 44 Go de VRAM cumulée, donc quelque chose
    # doit déborder.
    spec = _spec(tmp_path, "moe", hidden_size=4096, num_hidden_layers=60,
                 architectures=["Qwen3MoeForCausalLM"], num_experts=128,
                 num_experts_per_tok=8, moe_intermediate_size=1536)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    host_mlp = [lp for lp in plan.layers if lp.mlp_storage == "cpu"]
    host_attn = [lp for lp in plan.layers if lp.attn_storage == "cpu"]
    assert host_mlp, "ce modèle ne devrait pas tenir entièrement en VRAM"
    assert len(host_attn) < len(host_mlp)


def test_moe_reads_far_less_than_it_stores(tmp_path, target_rig):
    spec = _spec(tmp_path, "moe2", hidden_size=4096, num_hidden_layers=48,
                 architectures=["Qwen3MoeForCausalLM"], num_experts=128,
                 num_experts_per_tok=8, moe_intermediate_size=1536)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert plan.est_bytes_per_token < plan.total_weight_bytes / 4


def test_impossible_model_is_reported_not_silently_truncated(tmp_path, target_rig):
    spec = _spec(tmp_path, "huge", hidden_size=16384, intermediate_size=53248,
                 num_hidden_layers=126)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert any("ne tient pas sur cette machine" in w for w in plan.warnings)
    assert any("bits par poids" in w for w in plan.warnings)


def test_kv_budget_shrinks_to_keep_weights_in_vram(tmp_path, target_rig):
    spec = _spec(tmp_path, "seventy", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    greedy = plan_placement(spec, target_rig,
                            PlannerOptions(max_model_len=32768,
                                           max_concurrent_seqs=4,
                                           kv_vram_fraction=0.55))
    tuned, _ = auto_plan(spec, target_rig,
                         PlannerOptions(max_model_len=32768, max_concurrent_seqs=4))
    assert tuned.est_decode_tok_s > greedy.est_decode_tok_s


def test_plan_survives_a_json_round_trip(tmp_path, target_rig):
    spec = _spec(tmp_path, "rt")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert json.loads(json.dumps(plan.to_dict()))["model"] == plan.model
