"""Pièce 167 (poste3, 24/09) : `tests/test_gdn_int8_canal_153.py` ne teste que
`_est_projection_gdn`/`_est_projection_attn` EN ISOLATION -- aucun des 3 tests
n'appelle `convert_checkpoint`, donc rien ne garde l'INTÉGRATION de
`gdn_int8_canal` dans la boucle de conversion (`convert.py:1871-1874` et
`:2031-2033`). Preuve : en retirant la clause `gdn_int8_canal` à ces deux
points, les 3 tests restent verts (revue/poste3-piece167-24-09.md).

Ce test convertit RÉELLEMENT un mini-checkpoint (1 couche GDN, 1 couche
attention pleine, MLP, lm_head -- gabarit de test_collect_qwen35_gdn.py) sous
`gdn_int8_canal=True, attn_qkvo_int8_canal=True`, une promotion int8 forcée
(`snr_floor` haut, `promotion_classes` restreint à l'attention et au GDN) et
vérifie sur les OCTETS ÉCRITS (pas le manifeste, qui n'a pas de bilan GDN
comme `_bilan_attn_int8` pour l'attention) : self_attn et linear_attn en int8
PAR CANAL (`scales.shape[1] == 1`), MLP et lm_head INTACTS (jamais promus,
`promotion_classes` ne les cible pas)."""
import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from acvram.quant.convert import ConversionOptions, convert_checkpoint

# Pièce 167 : gabarit de test_collect_qwen35_gdn.py, dimensions ×8 (H=256, un
# multiple de 128) -- sur le gabarit d'origine (H=32), group_size=128 dépasse
# toujours la largeur du tenseur et COLLAPSE en ng=1 que la projection soit
# par canal ou par groupe : le bras cassant ne pouvait rien distinguer.
H, NH, NKV, HD = 256, 4, 2, 64                     # attention pleine (NH*HD = H)
NK, NV, DK, DV = 2, 4, 64, 64                       # GDN (Gated DeltaNet)
KEY_DIM, VALUE_DIM = NK * DK, NV * DV               # 128, 256
CONV_DIM = 2 * KEY_DIM + VALUE_DIM                  # 512
I, V, L = 384, 512, 2

# Suffixes des projections denses ciblées -- attention pleine ET GDN --
# jamais MLP ni lm_head : c'est CE tuple, pas le code sous test, qui garantit
# qu'eux seuls sont candidats à la promotion.
_CLASSES_ATTN_GDN = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "qkv", "gate", "alpha", "beta", "out")


def _checkpoint_167(tmp_path):
    """1 couche GDN (linear_attention), 1 couche attention pleine à porte
    (full_attention), MLP et lm_head -- noms BRUTS HF (`in_proj_*`, `A_log`),
    comme test_collect_qwen35_gdn.py, mais H=256 (voir la note ci-dessus)."""
    torch.manual_seed(20260924)
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
    sd[p0 + "conv1d.weight"] = r(CONV_DIM, 1, 4)
    sd[p0 + "dt_bias"] = r(NV)
    sd[p0 + "A_log"] = r(NV)
    sd[p0 + "norm.weight"] = r(DV)

    p1 = "model.layers.1.self_attn."
    sd[p1 + "q_proj.weight"] = r(NH * HD * 2, H)
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


def _convertir(tmp_path, gdn_int8_canal: bool):
    from acvram.engine.config import load_model_spec
    from acvram.hardware.profiles import load_profile
    from acvram.memory.tiering import PlannerOptions, auto_plan

    ckpt = _checkpoint_167(tmp_path)
    spec = load_model_spec(ckpt, "tiny-gdn-167")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    out = tmp_path / "out"
    convert_checkpoint(ckpt, plan, ConversionOptions(
        out_dir=str(out), attn_qkvo_int8_canal=True, gdn_int8_canal=gdn_int8_canal,
        snr_floor=999.0, promotion_classes=_CLASSES_ATTN_GDN, max_promotions=1.0), spec=spec)
    return out


def _scales_ng(out_dir, nom: str) -> int:
    """Nombre de groupes (`ng`) de l'échelle d'un tenseur int8 écrit ; 1 = par canal."""
    manifest = json.load(open(out_dir / "acvram_manifest.json"))
    fichier = manifest["weight_map"][f"{nom}.qweight"]
    with safe_open(out_dir / fichier, framework="pt", device="cpu") as fh:
        return fh.get_tensor(f"{nom}.scales").shape[1]


def _format(out_dir, nom: str) -> str:
    manifest = json.load(open(out_dir / "acvram_manifest.json"))
    return manifest["tensors"][nom]["format"]


def test_conversion_reelle_int8_canal_sur_attn_et_gdn_mlp_et_lm_head_intacts(tmp_path):
    out = _convertir(tmp_path, gdn_int8_canal=True)

    for nom in ("model.layers.0.linear_attn.qkv.weight", "model.layers.0.linear_attn.gate.weight",
               "model.layers.0.linear_attn.alpha.weight", "model.layers.0.linear_attn.beta.weight",
               "model.layers.0.linear_attn.out.weight"):
        assert _format(out, nom) == "int8", f"{nom} : pas promu int8"
        assert _scales_ng(out, nom) == 1, f"{nom} : pas par canal (ng={_scales_ng(out, nom)})"

    for nom in ("model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight",
               "model.layers.1.self_attn.v_proj.weight", "model.layers.1.self_attn.o_proj.weight"):
        assert _format(out, nom) == "int8", f"{nom} : pas promu int8"
        assert _scales_ng(out, nom) == 1, f"{nom} : pas par canal (ng={_scales_ng(out, nom)})"

    # MLP et lm_head : `promotion_classes` ne les cite pas -- jamais candidats,
    # donc jamais int8 (format d'origine, quel qu'il soit, mais PAS "int8").
    for nom in ("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
               "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.gate_proj.weight",
               "model.layers.1.mlp.up_proj.weight", "model.layers.1.mlp.down_proj.weight",
               "lm_head.weight"):
        assert _format(out, nom) != "int8", f"{nom} : promu int8 à tort"


def test_bras_casse_retirer_gdn_int8_canal_laisse_gdn_en_groupe(tmp_path, monkeypatch):
    """Pièce 167 : reproduit la faute nommée par chef (clause `gdn_int8_canal`
    retirée aux deux points d'intégration, convert.py:1871-1874 et :2031-2033)
    SANS toucher au fichier -- `_est_projection_gdn` neutralisée par
    monkeypatch dans le module `convert`, équivalent fonctionnel exact (les
    deux points d'intégration n'appellent QUE cette fonction pour le
    branchement GDN). Casse bien : les projections GDN retombent au groupe
    de 128 (ng > 1 sur ce gabarit, K=64 pour qkv)."""
    import acvram.quant.convert as conv
    monkeypatch.setattr(conv, "_est_projection_gdn", lambda nom: False)
    out = _convertir(tmp_path, gdn_int8_canal=True)
    ng = _scales_ng(out, "model.layers.0.linear_attn.qkv.weight")
    assert ng > 1, (
        f"le bras cassant n'a pas cassé : ng={ng} (attendu > 1, la faute aurait dû "
        f"faire retomber la projection GDN au groupe de {ConversionOptions(out_dir='').group_size})")
