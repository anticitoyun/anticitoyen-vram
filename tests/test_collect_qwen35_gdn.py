"""Calibration AWQ sur l'hybride GDN + attention pleine à porte (Qwen3.5),
chantier calibration-hybrides-gdn-17-09 (poste7-objectif-b1-17-09 §3 puis
verdict-qwen38-reconversion-17-09).

Qwen3.8-27B (`qwen3_5_text`) alterne des couches `linear_attention` (Gated
DeltaNet, 48/64 sur le vrai modèle) et `full_attention` à porte de sortie
(`attn_output_gate`, 16/64). Avant ce chantier : `_build_bf16_layer` n'avait
AUCUN chemin pour les couches GDN (KeyError attendu, structurel), et
l'attention pleine crashait en RuntimeError — `q_proj` sort deux fois
`num_attention_heads * head_dim`, pas une fois (`modeling_qwen3_5.py:
761-763`, chunké en query/porte à `:787-790`) — 0/64 couches calibrées au
total, confirmé par dry-run avant la reconversion `--no-awq` explicite.
"""
import json

import pytest
import torch
from safetensors.torch import save_file

# Pièce 267 : `acvram/engine/gdn.py:_refs()` importe transformers inconditionnellement
# (référence torch, même quand fla sert la voie réelle) — extra optionnel (`pyproject.toml`
# `gdn`), absent en CI de base (runner sans CUDA).
pytest.importorskip("transformers")

from acvram.engine.config import load_model_spec
from acvram.quant.collect import collect_activation_stats, load_calib_ids

H, NH, NKV, HD = 32, 4, 2, 8              # attention pleine
NK, NV, DK, DV = 2, 4, 8, 8                # GDN (Gated DeltaNet)
KEY_DIM, VALUE_DIM = NK * DK, NV * DV      # 16, 32
CONV_DIM = 2 * KEY_DIM + VALUE_DIM         # 64
I, V, L = 48, 64, 2


def _qwen35_checkpoint(tmp_path):
    """Une couche GDN (0), une couche attention pleine à porte (1) — les
    deux types du vrai modèle. Noms de tenseurs BRUTS HF (`in_proj_*`,
    `A_log`), pas encore renommés : c'est `collect_activation_stats` qui
    doit le faire, comme `convert.py::_adapt_hf` le fait pour le flux
    principal de conversion."""
    torch.manual_seed(20260917)
    d = tmp_path

    json.dump({
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "hidden_size": H, "intermediate_size": I,
        "num_hidden_layers": L, "num_attention_heads": NH,
        "num_key_value_heads": NKV, "head_dim": HD,
        "vocab_size": V, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
        "layer_types": ["linear_attention", "full_attention"],
        "attn_output_gate": True,
        "linear_num_key_heads": NK, "linear_num_value_heads": NV,
        "linear_key_head_dim": DK, "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": 4,
    }, open(d / "config.json", "w"))

    def r(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.02

    sd = {"model.embed_tokens.weight": r(V, H), "lm_head.weight": r(V, H)}

    p0 = "model.layers.0.linear_attn."
    sd[p0 + "in_proj_qkv.weight"] = r(CONV_DIM, H)
    sd[p0 + "in_proj_z.weight"] = r(VALUE_DIM, H)
    sd[p0 + "in_proj_a.weight"] = r(NV, H)
    sd[p0 + "in_proj_b.weight"] = r(NV, H)
    sd[p0 + "out_proj.weight"] = r(H, VALUE_DIM)
    sd[p0 + "conv1d.weight"] = r(CONV_DIM, 1, 4)        # [d, 1, L] brut HF, 3D
    sd[p0 + "dt_bias"] = r(NV)
    sd[p0 + "A_log"] = r(NV)
    sd[p0 + "norm.weight"] = r(DV)

    p1 = "model.layers.1.self_attn."
    sd[p1 + "q_proj.weight"] = r(NH * HD * 2, H)        # [q_h | porte_h] par tête
    sd[p1 + "k_proj.weight"] = r(NKV * HD, H)
    sd[p1 + "v_proj.weight"] = r(NKV * HD, H)
    sd[p1 + "o_proj.weight"] = r(H, NH * HD)
    sd[p1 + "q_norm.weight"] = r(HD)
    sd[p1 + "k_norm.weight"] = r(HD)

    for i in range(L):
        pm = f"model.layers.{i}.mlp."
        sd[pm + "gate_proj.weight"] = r(I, H)
        sd[pm + "up_proj.weight"] = r(I, H)
        sd[pm + "down_proj.weight"] = r(H, I)
        pn = f"model.layers.{i}."
        # convention HF Qwen3.5 : normes centrées à zéro, (1+w) à la lecture
        # (`_NORMES_ZERO_CENTREES`) — tirées petites, PAS à 1, pour que le
        # second test distingue "avec le +1" de "sans".
        sd[pn + "input_layernorm.weight"] = r(H)
        sd[pn + "post_attention_layernorm.weight"] = r(H)
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


def test_calibration_couvre_les_deux_couches_gdn_et_pleine(tmp_path):
    """64/64 couches sur le vrai modèle, ici 2/2 sur le jouet — le critère
    du chantier calibration-hybrides-gdn-17-09."""
    from acvram.server.chat import load_tokenizer

    ckpt = _qwen35_checkpoint(tmp_path)
    spec = load_model_spec(ckpt, "qwen35")
    assert spec.layer_types == ["linear_attention", "full_attention"]
    assert spec.attn_output_gate
    tok = load_tokenizer(ckpt)
    calib = load_calib_ids(tok, None, 4, 24, spec.vocab_size)

    stats = collect_activation_stats(ckpt, spec, calib, device="cpu",
                                     dtype=torch.float32)

    attendus = [
        "model.layers.0.linear_attn.qkv.weight",
        "model.layers.0.linear_attn.gate.weight",
        "model.layers.0.linear_attn.alpha.weight",
        "model.layers.0.linear_attn.beta.weight",
        "model.layers.0.linear_attn.out.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ]
    assert len(attendus) == 15, "13 lineaires calibrables sur 2 couches (5 GDN+3 mlp, 4 attn+3 mlp)"
    for nom in attendus:
        assert nom in stats, f"couche non calibrée : {nom}"
        assert torch.isfinite(stats[nom].mean_abs).all(), nom
        assert stats[nom].n_samples > 0, nom


def test_les_normes_zero_centrees_sont_relevees_a_1_plus_w(tmp_path):
    """Témoin (REGLES §5) : sans le `+1`, les normes valent ~0 (poids
    tirés autour de zéro dans le point de contrôle brut) et la sortie
    normée est ~0 partout — les statistiques de la première linéaire de
    chaque couche seraient quasi nulles au lieu de porter le signal."""
    from acvram.server.chat import load_tokenizer

    ckpt = _qwen35_checkpoint(tmp_path)
    from safetensors import safe_open
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt", device="cpu") as fh:
        brut = fh.get_tensor("model.layers.0.input_layernorm.weight")
    assert brut.abs().mean() < 0.1, "le poids brut doit être proche de zéro (convention HF)"

    spec = load_model_spec(ckpt, "qwen35")
    tok = load_tokenizer(ckpt)
    calib = load_calib_ids(tok, None, 4, 24, spec.vocab_size)
    stats = collect_activation_stats(ckpt, spec, calib, device="cpu",
                                     dtype=torch.float32)

    # Entrée de la première linéaire de chaque couche = sortie de
    # input_layernorm : proche de 1 en magnitude si (1+w) est bien lu
    # (RMS-normalisée puis multipliée par un poids proche de 1), proche de
    # 0,02 (le poids brut) sinon.
    assert stats["model.layers.0.linear_attn.qkv.weight"].mean_abs.mean() > 0.3
    assert stats["model.layers.1.self_attn.q_proj.weight"].mean_abs.mean() > 0.3
