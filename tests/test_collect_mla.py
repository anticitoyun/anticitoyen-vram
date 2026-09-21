"""Calibration AWQ sur un modele MLA (bead : signalement Manon/Jerome, 15/09).

GLM-4.7-Flash (q_lora_rank>0, kv_lora_rank>0) n'a pas de `self_attn.q_proj`
plat -- `_build_bf16_layer` (collect.py) le supposait pourtant, et le
KeyError remontait jusqu'a desactiver AWQ pour TOUT le modele (cli.py:469),
pour une seule couche fautive. Corrige par structure (`spec.est_mla`),
comme le bead anticitoyen-vram-992 l'avait fait pour loader.py/convert.py.
"""

import json

import torch
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.quant.collect import collect_activation_stats, load_calib_ids


def _mla_checkpoint(tmp_path, casser_une_couche=False):
    """Deux couches MLA+MoE minuscules : q_lora, KV compresse, routage
    sigmoid+biais (GLM-4.7-Flash), expert partage. `casser_une_couche`
    supprime `q_a_proj` de la couche 1 SEULE, pour verifier la resilience
    par couche."""
    torch.manual_seed(20260915)
    H, NH, L, V = 64, 4, 2, 512
    QLORA, KVLORA, ROPE, NOPE, VDIM = 48, 32, 8, 16, 16
    NE, TOPK, MOE_I = 4, 2, 32
    d = tmp_path

    json.dump({
        "model_type": "glm4_moe_lite",
        "hidden_size": H, "intermediate_size": 128,
        "num_hidden_layers": L, "num_attention_heads": NH,
        "num_key_value_heads": NH, "vocab_size": V,
        "max_position_embeddings": 512,
        "kv_lora_rank": KVLORA, "q_lora_rank": QLORA,
        "qk_rope_head_dim": ROPE, "qk_nope_head_dim": NOPE, "v_head_dim": VDIM,
        "num_experts": NE, "num_experts_per_tok": TOPK,
        "moe_intermediate_size": MOE_I, "router_scoring": "sigmoid",
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))

    def r(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.02

    sd = {"model.embed_tokens.weight": r(V, H), "lm_head.weight": r(V, H),
         "model.norm.weight": torch.ones(H, dtype=torch.bfloat16)}
    for i in range(L):
        p = f"model.layers.{i}."
        if not (casser_une_couche and i == 1):
            sd[p + "self_attn.q_a_proj.weight"] = r(QLORA, H)
            sd[p + "self_attn.q_a_layernorm.weight"] = torch.ones(QLORA, dtype=torch.bfloat16)
            sd[p + "self_attn.q_b_proj.weight"] = r(NH * (NOPE + ROPE), QLORA)
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = r(KVLORA + ROPE, H)
        sd[p + "self_attn.kv_a_layernorm.weight"] = torch.ones(KVLORA, dtype=torch.bfloat16)
        sd[p + "self_attn.o_proj.weight"] = r(H, NH * VDIM)
        sd[p + "mlp.gate.weight"] = r(NE, H)
        sd[p + "mlp.gate.e_score_correction_bias"] = torch.zeros(NE, dtype=torch.float32)
        for e in range(NE):
            ep = p + f"mlp.experts.{e}."
            sd[ep + "gate_proj.weight"] = r(MOE_I, H)
            sd[ep + "up_proj.weight"] = r(MOE_I, H)
            sd[ep + "down_proj.weight"] = r(H, MOE_I)
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    save_file(sd, str(d / "model.safetensors"))

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {f"tok{i}": i for i in range(V)}
    for i, w in enumerate(["the", "a", "of", "and", "quick", "brown", "fox"]):
        vocab[w] = 400 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(str(d / "tokenizer.json"))
    json.dump({"eos_token": "tok0"}, open(d / "tokenizer_config.json", "w"))
    return str(d)


def test_calibration_mla_ne_leve_pas_et_couvre_les_bons_tenseurs(tmp_path):
    from acvram.server.chat import load_tokenizer

    ckpt = _mla_checkpoint(tmp_path)
    spec = load_model_spec(ckpt, "mla")
    assert spec.est_mla
    tok = load_tokenizer(ckpt)
    calib = load_calib_ids(tok, None, 4, 32, spec.vocab_size)

    stats = collect_activation_stats(ckpt, spec, calib, device="cpu",
                                     dtype=torch.float32)

    for i in range(spec.num_layers):
        p = f"model.layers.{i}.self_attn."
        for nom in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            assert p + nom + ".weight" in stats, f"manquant : {p}{nom}.weight"


def test_une_couche_cassee_ne_perd_que_ses_statistiques(tmp_path, capsys):
    """Sans la resilience par couche, ce test echouerait avec un KeyError
    non rattrape et `stats` resterait vide (le mecanisme de cli.py:469
    desactiverait AWQ pour tout le modele)."""
    from acvram.server.chat import load_tokenizer

    ckpt = _mla_checkpoint(tmp_path, casser_une_couche=True)
    spec = load_model_spec(ckpt, "mla")
    tok = load_tokenizer(ckpt)
    calib = load_calib_ids(tok, None, 4, 32, spec.vocab_size)

    stats = collect_activation_stats(ckpt, spec, calib, device="cpu",
                                     dtype=torch.float32)

    assert "model.layers.0.self_attn.q_a_proj.weight" in stats
    assert "model.layers.1.self_attn.q_a_proj.weight" not in stats
    assert "avertissement" in capsys.readouterr().out
