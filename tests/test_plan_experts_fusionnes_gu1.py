"""gu1 (poste1, 02/10) : `acvram plan` sur une source HF à experts FUSIONNÉS (gpt-oss, MXFP4) sous-comptait la couche.
`_affiner_couches` prenait le suffixe qui suit « .experts. » pour un numéro d'expert — gate_up_proj_blocks,
gate_up_proj_scales, gate_up_proj_bias, down_proj_… : 6 « experts » au lieu de 128 — et comptait un octet MXFP4 pour
un paramètre (il en porte deux), échelles E8M0 comprises. Le plan du 120b annonçait 36,3 Go, et « tient sur deux
cartes sans RAM hôte » avec --gpus all, pour ≈ 68,8 Go réels ; celui du 20b source 7,56 Go contre 12,6 sur le converti.
Les en-têtes ci-dessous sont ceux du vrai gpt-oss-120b (couche 0, relevés le 02/10), sans un octet de poids.
Cassures éprouvées : suffixe repris comme numéro d'expert → test_experts_comptes rouge ; octet = un paramètre et
échelles comptées → test_parametres_de_la_couche et test_plan_120b_… rouges (le plan retombe à 36,3 Go, le symptôme)."""
import json
import struct

from acvram.engine.config import load_model_spec

H, I, E, NB = 2880, 2880, 128, 90

CONFIG = {
    "architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "attention_bias": True, "head_dim": 64,
    "hidden_size": H, "intermediate_size": I, "num_hidden_layers": 36, "num_attention_heads": 64,
    "num_key_value_heads": 8, "num_local_experts": E, "num_experts_per_tok": 4, "experts_per_token": 4,
    "vocab_size": 201088, "max_position_embeddings": 131072, "rms_norm_eps": 1e-5, "rope_theta": 150000,
    "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
                     "rope_type": "yarn", "truncate": False},
    "sliding_window": 128, "swiglu_limit": 7.0, "tie_word_embeddings": False,
    "layer_types": ["sliding_attention", "full_attention"] * 18,
    "quantization_config": {"quant_method": "mxfp4"},
}


def _couche(i):
    p = f"model.layers.{i}."
    return {
        p + "input_layernorm.weight": ("BF16", [H]),
        p + "post_attention_layernorm.weight": ("BF16", [H]),
        p + "mlp.experts.down_proj_bias": ("BF16", [E, H]),
        p + "mlp.experts.down_proj_blocks": ("U8", [E, H, NB, 16]),
        p + "mlp.experts.down_proj_scales": ("U8", [E, H, NB]),
        p + "mlp.experts.gate_up_proj_bias": ("BF16", [E, 2 * I]),
        p + "mlp.experts.gate_up_proj_blocks": ("U8", [E, 2 * I, NB, 16]),
        p + "mlp.experts.gate_up_proj_scales": ("U8", [E, 2 * I, NB]),
        p + "mlp.router.bias": ("BF16", [E]),
        p + "mlp.router.weight": ("BF16", [E, H]),
        p + "self_attn.q_proj.weight": ("BF16", [4096, H]), p + "self_attn.q_proj.bias": ("BF16", [4096]),
        p + "self_attn.k_proj.weight": ("BF16", [512, H]), p + "self_attn.k_proj.bias": ("BF16", [512]),
        p + "self_attn.v_proj.weight": ("BF16", [512, H]), p + "self_attn.v_proj.bias": ("BF16", [512]),
        p + "self_attn.o_proj.weight": ("BF16", [H, 4096]), p + "self_attn.o_proj.bias": ("BF16", [H]),
        p + "self_attn.sinks": ("BF16", [64]),
    }


def _source(tmp_path):
    """Dépôt HF dont les fragments n'ont que leur en-tête : le plan ne lit que les formes."""
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    tenseurs = {"model.embed_tokens.weight": ("BF16", [201088, H]), "lm_head.weight": ("BF16", [201088, H]),
                "model.norm.weight": ("BF16", [H])}
    for i in range(36):
        tenseurs.update(_couche(i))
    entete = {n: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for n, (d, s) in tenseurs.items()}
    brut = json.dumps(entete).encode()
    (tmp_path / "model-00000-of-00001.safetensors").write_bytes(struct.pack("<Q", len(brut)) + brut)
    return str(tmp_path)


def test_experts_comptes(tmp_path):
    l = load_model_spec(_source(tmp_path)).layers[0]
    assert l.is_moe and l.n_experts == E and l.n_experts_active == 4


def test_parametres_de_la_couche(tmp_path):
    l = load_model_spec(_source(tmp_path)).layers[0]
    experts = E * (2 * I * H + H * I)                        # deux codes E2M1 par octet, échelles exclues
    biais_et_routeur = E * (2 * I + H) + E * H + E
    assert l.mlp_params == experts + biais_et_routeur
    assert l.attn_params == 4096 * H + 2 * 512 * H + H * 4096 + 4096 + 2 * 512 + H + 64


def test_plan_120b_n_tient_pas_sur_la_carte(tmp_path):
    """Le symptôme : ≈ 66 Go au format NVFP4, donc pas sur les seules cartes du rig (32 + 12 Gio)."""
    from acvram.hardware.detect import detect_rig
    from acvram.memory.tiering import PlannerOptions, auto_plan
    spec = load_model_spec(_source(tmp_path))
    plan, _ = auto_plan(spec, detect_rig("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=8192, max_concurrent_seqs=1, force_format="nvfp4"))
    assert 60e9 < plan.total_weight_bytes < 72e9, plan.total_weight_bytes
    assert plan.bytes_per_tier.get("cpu", 0) > 20e9
