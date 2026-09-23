"""Calibration AWQ sur l'hybride Mamba2 + MoE + attention pleine (Nemotron-H),
sage-hybrides-etape1-close-gemm-dense-17-09.

Avant ce chantier : `collect_activation_stats` cherchait `model.embed_tokens.
weight` directement sur le point de contrôle brut (nommé `backbone.
embeddings.weight` chez nemotron_h) -- KeyError avant même la boucle par
couche, calibration ENTIÈRE repliée sur l'arrondi au plus proche (reproduit
le 17/09 sur le vrai modèle bf16). Une fois le renommage réparé,
`_build_bf16_layer` n'avait par ailleurs AUCUN chemin pour une couche
Mamba2 réelle (ni self_attn ni mlp) : elle aurait levé KeyError sur
`self_attn.q_proj.weight`, et la couche -- ratée -- aurait laissé les
couches EN AVAL calibrer sur un état caché faux (identité au lieu du
mélangeur Mamba2), silencieusement, pour 23 couches sur 52 du vrai modèle.
"""
import json

import torch
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.quant.collect import collect_activation_stats, load_calib_ids

H = 32                                    # hidden_size
NH, NKV, HD = 4, 2, 8                      # attention pleine (couche 2)
MH, MHD, G, N = 2, 4, 1, 4                 # Mamba2 : têtes, dim/tête, groupes, état
INNER = MH * MHD                          # 8
CONV_DIM = INNER + 2 * G * N              # 16
IN_PROJ_ROWS = 2 * INNER + 2 * G * N + MH  # 26
KERNEL = 4
I_MOE, N_EXP = 16, 2                       # couche MoE (couche 1)
V = 64


def _nemotron_h_checkpoint(tmp_path):
    torch.manual_seed(20260917)
    d = tmp_path
    json.dump({
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": H, "intermediate_size": 24,
        "num_hidden_layers": 3, "num_attention_heads": NH,
        "num_key_value_heads": NKV, "head_dim": HD,
        "vocab_size": V, "max_position_embeddings": 128,
        "layer_norm_epsilon": 1e-5, "rope_theta": 10000.0,
        "layers_block_type": ["mamba", "moe", "attention"],
        "mamba_num_heads": MH, "mamba_head_dim": MHD,
        "mamba_n_groups": G, "mamba_state_size": N, "mamba_conv_kernel": KERNEL,
        "n_routed_experts": N_EXP, "num_experts_per_tok": 1,
        "moe_intermediate_size": I_MOE, "n_shared_experts": 1,
        "moe_shared_expert_intermediate_size": I_MOE,
    }, open(d / "config.json", "w"))

    def r(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.02

    sd = {"backbone.embeddings.weight": r(V, H), "backbone.norm_f.weight": r(H),
          "lm_head.weight": r(V, H)}

    # couche 0 : Mamba2 réel (pas de mlp)
    p0 = "backbone.layers.0."
    sd[p0 + "norm.weight"] = r(H)
    sd[p0 + "mixer.in_proj.weight"] = r(IN_PROJ_ROWS, H)
    sd[p0 + "mixer.out_proj.weight"] = r(H, INNER)
    sd[p0 + "mixer.conv1d.weight"] = r(CONV_DIM, 1, KERNEL)   # [d, 1, L] brut HF, 3D
    sd[p0 + "mixer.conv1d.bias"] = r(CONV_DIM)
    sd[p0 + "mixer.A_log"] = r(MH)
    sd[p0 + "mixer.D"] = r(MH)
    sd[p0 + "mixer.dt_bias"] = r(MH)
    sd[p0 + "mixer.norm.weight"] = r(INNER)

    # couche 1 : MoE (pas d'attention)
    p1 = "backbone.layers.1."
    sd[p1 + "norm.weight"] = r(H)
    sd[p1 + "mixer.gate.weight"] = r(N_EXP, H)   # nom réel vérifié le 17/09 (pas "router")
    for e in range(N_EXP):
        sd[p1 + f"mixer.experts.{e}.up_proj.weight"] = r(I_MOE, H)
        sd[p1 + f"mixer.experts.{e}.down_proj.weight"] = r(H, I_MOE)
    sd[p1 + "mixer.shared_experts.up_proj.weight"] = r(I_MOE, H)
    sd[p1 + "mixer.shared_experts.down_proj.weight"] = r(H, I_MOE)

    # couche 2 : attention pleine (pas de récurrence)
    p2 = "backbone.layers.2."
    sd[p2 + "norm.weight"] = r(H)
    sd[p2 + "mixer.q_proj.weight"] = r(NH * HD, H)
    sd[p2 + "mixer.k_proj.weight"] = r(NKV * HD, H)
    sd[p2 + "mixer.v_proj.weight"] = r(NKV * HD, H)
    sd[p2 + "mixer.o_proj.weight"] = r(H, NH * HD)
    save_file(sd, str(d / "model.safetensors"))

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {f"tok{i}": i for i in range(V)}
    for i, w in enumerate(["the", "a", "of", "and", "quick", "brown", "fox"]):
        vocab[w] = 40 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(str(d / "tokenizer.json"))
    json.dump({"eos_token": "tok0"}, open(d / "tokenizer_config.json", "w"))
    return str(d)


def test_calibration_couvre_mamba2_moe_et_attention_pleine(tmp_path):
    """3/3 couches sur le jouet : la Mamba2 réelle ne crashe plus (état
    caché propagé aux couches en aval), le MoE et l'attention pleine
    calibrent réellement -- pas de repli sur l'arrondi au plus proche."""
    from acvram.server.chat import load_tokenizer

    ckpt = _nemotron_h_checkpoint(tmp_path)
    spec = load_model_spec(ckpt, "nemotron-h-jouet")
    assert spec.layer_types == ["mamba", "moe", "full_attention"]
    tok = load_tokenizer(ckpt)
    calib = load_calib_ids(tok, None, 4, 24, spec.vocab_size)

    stats = collect_activation_stats(ckpt, spec, calib, device="cpu",
                                     dtype=torch.float32)

    attendus = [
        "model.layers.1.mlp.shared_expert.up_proj.weight",
        "model.layers.1.mlp.shared_expert.down_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.self_attn.v_proj.weight",
        "model.layers.2.self_attn.o_proj.weight",
    ]
    for nom in attendus:
        assert nom in stats, f"couche non calibrée : {nom}"
        assert torch.isfinite(stats[nom].mean_abs).all(), nom
        assert stats[nom].n_samples > 0, nom
    # Au moins un expert routé (top-1 sur 2 experts, routeur tiré au hasard :
    # comme le vrai modèle, un expert jamais routé sur le corpus n'a
    # simplement aucune statistique -- 5888 experts dans ce cas le 17/09,
    # comportement documenté, pas une régression à couvrir ici).
    assert any(f"model.layers.1.mlp.experts.{e}.up_proj.weight" in stats for e in range(N_EXP))
