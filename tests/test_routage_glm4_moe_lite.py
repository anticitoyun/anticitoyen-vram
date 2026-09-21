"""Réfutation de Sage (14/09, revue/sage-refutation-glm-routage-14-09.md) :
`glm4_moe_lite` (GLM-4.7-Flash) route en sigmoid + biais de correction de
façon INCONDITIONNELLE côté HF (modeling_glm4_moe_lite.py:402-423), mais
sa config ne déclare pas `scoring_func` — le repli générique de config.py
retombait sur "softmax", qui ignore le biais et se trompe sur un
sous-ensemble des jetons (8/16 mesurés, équivalence CPU 2 couches vs HF).
"""
import json

from acvram.engine.config import load_model_spec


def _config(tmp_path, model_type, **extra):
    cfg = {
        "model_type": model_type,
        "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 4, "vocab_size": 1000,
        "max_position_embeddings": 512,
        "kv_lora_rank": 512, "q_lora_rank": 384,
        "qk_rope_head_dim": 32, "qk_nope_head_dim": 96, "v_head_dim": 128,
        "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        **extra,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def test_glm4_moe_lite_route_en_sigmoid_sans_scoring_func_declare(tmp_path):
    """Le cas réfuté : aucune clé `scoring_func`/`router_scoring` dans la
    config — un changement qui doit casser : remettre le repli générique
    seul (sans le cas `glm4_moe_lite` explicite) romprait ce test."""
    spec = load_model_spec(_config(tmp_path, "glm4_moe_lite"))
    assert spec.router_scoring == "sigmoid"


def test_ernie_garde_softmax_malgre_le_biais(tmp_path):
    """Pas un heuristique « sigmoid si biais » : ernie4_5_moe a un biais de
    correction ET reste en softmax, légitimement (config.py, ~ligne 534)."""
    spec = load_model_spec(_config(tmp_path, "ernie4_5_moe",
                                   moe_num_experts=4, moe_k=2))
    assert spec.router_scoring == "softmax"


def test_glm4_moe_garde_son_repli_generique(tmp_path):
    """`glm4_moe` (sans `_lite`) n'est pas touché par ce cas spécifique —
    il suit le repli générique (`scoring_func`/`router_scoring` déclaré ou
    "softmax")."""
    spec = load_model_spec(_config(tmp_path, "glm4_moe"))
    assert spec.router_scoring == "softmax"


def test_scoring_func_explicite_l_emporte_sur_glm4_moe_lite(tmp_path):
    """Si la config déclarait un jour `scoring_func`/`router_scoring`
    explicitement pour cette famille, le cas spécial ne doit pas l'écraser
    — ce n'est pas le cas mesuré aujourd'hui (HF ne déclare rien), mais le
    câblage ne doit pas dépendre de cette absence pour rester correct."""
    spec = load_model_spec(_config(tmp_path, "glm4_moe_lite",
                                   router_scoring="softmax"))
    assert spec.router_scoring == "sigmoid", (
        "glm4_moe_lite doit rester en sigmoid : c'est un fait du modèle "
        "(HF, inconditionnel), pas une préférence de conversion à respecter")
