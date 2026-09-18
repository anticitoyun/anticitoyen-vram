"""Devstral-Small-2-24B (chef, 18/09) : Mistral3ForConditionalGeneration,
text_config ministral3, yarn dans rope_parameters (pas rope_scaling)."""
import json
import math

from acvram.engine.config import load_model_spec

ROPE_PARAMETERS = {
    "type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 16384,
    "max_position_embeddings": 262144,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale_all_dim": 1.0,
    "mscale": 1.0,
    "llama_4_scaling_beta": 0.1,
}


def _devstral_config(tmp_path):
    d = tmp_path / "devstral"
    d.mkdir()
    json.dump({
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": {
            "model_type": "ministral3",
            "hidden_size": 5120, "intermediate_size": 32768,
            "num_hidden_layers": 40, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 131072, "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5, "rope_parameters": ROPE_PARAMETERS,
        },
    }, open(d / "config.json", "w"))
    return str(d)


def test_mistral3_route_vers_llama():
    """archs[0] doit trouver l'alias explicite, pas seulement le repli par
    defaut -- un repli qui marche aujourd'hui casse sans bruit si le defaut
    de `_ARCH_ALIASES.get(archs[0], "llama")` change un jour."""
    from acvram.engine.config import _ARCH_ALIASES
    assert _ARCH_ALIASES["Mistral3ForConditionalGeneration"] == "llama"


def test_devstral_deplie_text_config_et_porte_le_yarn(tmp_path):
    spec = load_model_spec(_devstral_config(tmp_path), "devstral")
    assert spec.architecture == "llama"
    assert spec.hidden_size == 5120
    assert spec.num_layers == 40
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.rope_scaling is not None
    assert spec.rope_scaling.get("type") == "yarn"
    assert spec.rope_scaling.get("factor") == 16.0
    assert spec.rope_scaling.get("original_max_position_embeddings") == 16384
    assert spec.rope_scaling.get("beta_fast") == 32.0
    assert spec.rope_scaling.get("beta_slow") == 1.0


def test_temoin_cassant_sans_le_branchement_rope_scaling_reste_none(tmp_path, monkeypatch):
    """Bras casse (REGLES §5) : neutralise la branche ministral3 -- rope_
    scaling doit retomber a None, prouvant que c'est bien ELLE qui portait
    le yarn, pas un autre chemin deja generique."""
    import acvram.engine.config as config_mod
    original = config_mod.load_model_spec
    src = _devstral_config(tmp_path)
    cfg_path = src + "/config.json"
    cfg = json.load(open(cfg_path))
    cfg["text_config"]["model_type"] = "ministral3_temoin_absent"
    json.dump(cfg, open(cfg_path, "w"))
    spec = original(src, "devstral")
    assert spec.rope_scaling is None


def test_llama_4_scaling_beta_vaut_1_sous_16384_positions():
    """Verifie numeriquement la formule HF (modeling_ministral3.py,
    get_llama_4_attn_scale) qui justifie de NE PAS porter ce champ :
    floor(position/original_max_position_embeddings) = 0 pour toute
    position < 16384, donc scaling = 1 + beta*log(1) = 1 exactement."""
    beta, orig = 0.1, 16384
    for position in (0, 1, 8191, 8192, 16383):
        scaling = 1 + beta * math.log(1 + math.floor(position / orig))
        assert scaling == 1.0, f"position {position} : scaling {scaling} != 1.0"
    # temoin positif : au-dela, la formule bouge bien (sinon le calcul lui-meme serait mort)
    scaling_au_dela = 1 + beta * math.log(1 + math.floor(16384 / orig))
    assert scaling_au_dela != 1.0
