"""sage-hybrides-etape1-close-gemm-dense-17-09 : Nemotron-3.5-30B-A3B converti
par acvram mesure 1,0632 contre 0,987 pour le checkpoint NVFP4 officiel
(`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Nemotron-3.5-Lightning-30B-A3B-NVFP4/
hf_quant_config.json`, quant_algo=MIXED_PRECISION). Comparaison dtype tenseur
par tenseur (20 min, à sec) : ce fichier NE liste QUE `mixer.experts.*`
(NVFP4) et `mixer.shared_experts.*` (NVFP4) en plus de `mixer.in_proj`/
`out_proj` (FP8, 23 couches Mamba2) -- aucune entrée `self_attn` sur les 6
couches full_attention, donc laissées en bf16. Notre conversion précédente
quantifiait TOUT en NVFP4, y compris ces deux familles.

acvram ne porte pas de format FP8 autonome pour les poids (seulement comme
échelle de bloc E4M3 à l'intérieur de NVFP4/Q3N) : bf16 est la meilleure
approximation disponible pour `mixer.in_proj`/`out_proj`, plus fine que le
FP8 officiel, jamais plus grossière.
"""
from collections import namedtuple

from acvram.engine.config import ModelSpec
from acvram.quant.convert import ConversionOptions, TensorRouter

_FakeLayerPlan = namedtuple("_FakeLayerPlan", "index fmt")
_FakePlan = namedtuple("_FakePlan", "layers")


def _router_nemotron_h(num_layers=6):
    spec = ModelSpec(
        name="nemotron-test", model_type="nemotron_h", architecture="llama",
        hidden_size=128, intermediate_size=256, num_layers=num_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=1000, max_position_embeddings=512, rope_theta=10000.0,
        layer_types=(["mamba", "moe", "mamba", "moe", "mamba", "full_attention"]
                     * ((num_layers // 6) + 1))[:num_layers],
    )
    plan = _FakePlan(layers=[_FakeLayerPlan(index=i, fmt="nvfp4") for i in range(num_layers)])
    return TensorRouter(spec, plan, ConversionOptions(out_dir="/tmp"))


def test_mamba_in_out_proj_reste_16_bits_pas_nvfp4():
    """Un changement qui doit casser : retirer la branche `model_type ==
    "nemotron_h"` de `format_for` remet ces deux tenseurs en `nvfp4` (le
    format de la couche) comme avant ce correctif."""
    router = _router_nemotron_h()
    assert router.format_for("model.layers.0.mamba.in_proj.weight") == "bf16"
    assert router.format_for("model.layers.0.mamba.out_proj.weight") == "bf16"


def test_self_attn_des_couches_full_attention_reste_16_bits():
    router = _router_nemotron_h()
    for suffixe in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
        assert router.format_for(f"model.layers.5.self_attn.{suffixe}") == "bf16"


def test_les_experts_moe_restent_quantifies():
    """La correction ne doit PAS s'étendre aux experts MoE : le checkpoint
    officiel les garde en NVFP4 (`W4A16_NVFP4`), seuls `mixer.in_proj`/
    `out_proj` et `self_attn` en sortent."""
    router = _router_nemotron_h()
    assert router.format_for("model.layers.1.mlp.experts.0.down_proj.weight") == "nvfp4"
    assert router.format_for("model.layers.1.mlp.experts.0.up_proj.weight") == "nvfp4"


def test_hors_nemotron_h_lattention_reste_quantifiee():
    """Le correctif est architecture-spécifique : un modèle dense ordinaire
    (`model_type` différent de `nemotron_h`) garde son attention quantifiée
    comme avant -- sinon la correction casserait tous les modèles denses
    existants."""
    spec = ModelSpec(
        name="llama-test", model_type="llama", architecture="llama",
        hidden_size=128, intermediate_size=256, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        vocab_size=1000, max_position_embeddings=512, rope_theta=10000.0,
    )
    plan = _FakePlan(layers=[_FakeLayerPlan(index=i, fmt="nvfp4") for i in range(2)])
    router = TensorRouter(spec, plan, ConversionOptions(out_dir="/tmp"))
    assert router.format_for("model.layers.0.self_attn.q_proj.weight") == "nvfp4"
