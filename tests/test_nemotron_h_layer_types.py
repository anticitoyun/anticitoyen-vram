"""Le point de contrôle bf16 source de Nemotron-3.5-30B-A3B (avant toute
double quantification) porte `layers_block_type` (liste de mots) mais PAS
`hybrid_override_pattern` (motif M/E/-/*) — ce dernier n'apparaît que sur le
dérivé EXL3 déjà mesuré (1,0795, condamné). `load_model_spec` ne lisait que
`hybrid_override_pattern` : sur ce bf16, `layer_types` restait `None`,
`couches_recurrentes` valait 0, et la conversion réclamait `self_attn.q_proj`
sur les 23 couches Mamba2 qui n'en portent pas — `ValueError: 46 tenseurs
attendus absents (premier : model.layers.0.self_attn.q_proj.weight)`,
reproduit tel quel le 17/09 par un dry-run réel sur le point de contrôle."""
import json

from acvram.engine.config import load_model_spec

_LAYERS_BLOCK_TYPE = (
    ["mamba", "moe", "mamba", "moe", "mamba", "attention"] * 8
    + ["mamba", "moe", "mamba", "moe"]
)  # 52 entrées, même proportions (23 mamba / 23 moe / 6 attention) que le vrai config.json


def _config_nemotron_h(tmp_path, avec_motif=False):
    cfg = {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": len(_LAYERS_BLOCK_TYPE), "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 1000,
        "max_position_embeddings": 512,
        "n_routed_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64, "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": 64,
        "layers_block_type": _LAYERS_BLOCK_TYPE,
    }
    if avec_motif:
        cfg["hybrid_override_pattern"] = "".join(
            {"mamba": "M", "moe": "E", "attention": "*"}[t] for t in _LAYERS_BLOCK_TYPE)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def test_layers_block_type_seul_donne_les_bons_types_de_couche(tmp_path):
    """Un changement qui doit casser : retirer la branche `else` (celle qui
    lit `layers_block_type`) et ne garder que `hybrid_override_pattern`
    remet `layer_types` à `None` sur ce cas."""
    spec = load_model_spec(_config_nemotron_h(tmp_path, avec_motif=False))
    assert spec.layer_types is not None
    assert spec.layer_types[:6] == [
        "mamba", "moe", "mamba", "moe", "mamba", "full_attention"]
    assert spec.couches_recurrentes == sum(
        1 for t in _LAYERS_BLOCK_TYPE if t == "mamba")


def test_hybrid_override_pattern_prime_quand_present(tmp_path):
    """Même résultat que `layers_block_type` sur ce jeu de données — les deux
    champs décrivent la même chose, `hybrid_override_pattern` reste la
    branche prioritaire quand les deux sont là (dérivés EXL3)."""
    spec_motif = load_model_spec(_config_nemotron_h(tmp_path, avec_motif=True))
    spec_mots = load_model_spec(_config_nemotron_h(tmp_path, avec_motif=False))
    assert spec_motif.layer_types == spec_mots.layer_types
