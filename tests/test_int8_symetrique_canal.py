"""poste7-p2-qkvo-int8-canal-18-09 : poste7 demande, pour le P2 cuBLASLt/
torch._int_mm, un int8 SYMETRIQUE PAR CANAL (une echelle par ligne, sans
point-zero variable) sur q/k/v/o uniquement -- le reste du modele garde le
groupe de 128 affine. Deux niveaux de controle : le format lui-meme
(formats.py, sans passer par une conversion complete), et le cablage dans
convert_checkpoint (ConversionOptions.attn_qkvo_int8_canal, groute vers q/k/
v/o seulement quand le routeur les a places en int8)."""
import json

import torch
from safetensors.torch import save_file

from acvram.quant.convert import _est_projection_attn
from acvram.quant.formats import INT8Tensor, _dequantize_int8, _quantize_int8


def test_projection_attn_couvre_gqa_et_mla_pas_les_normes():
    """18/09, chantier GLM (MLA) : q/k/v/o ne nomme rien chez GLM
    (q_a_proj/q_b_proj/kv_a_proj_with_mqa/o_proj) -- `_est_projection_attn`
    doit couvrir les deux architectures sans suffixe fixe, et exclure les
    normes (q_a_layernorm/kv_a_layernorm), jamais candidates a l'int8."""
    for nom in ("model.layers.0.self_attn.q_proj.weight",
               "model.layers.0.self_attn.k_proj.weight",
               "model.layers.0.self_attn.v_proj.weight",
               "model.layers.0.self_attn.o_proj.weight",
               "model.layers.0.self_attn.q_a_proj.weight",
               "model.layers.0.self_attn.q_b_proj.weight",
               "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
               "model.layers.0.self_attn.kv_b_proj.weight"):
        assert _est_projection_attn(nom), f"projection non reconnue : {nom}"
    for nom in ("model.layers.0.self_attn.q_a_layernorm.weight",
               "model.layers.0.self_attn.kv_a_layernorm.weight",
               "model.layers.0.mlp.gate_proj.weight",
               "model.norm.weight"):
        assert not _est_projection_attn(nom), f"faux positif : {nom}"


def _petit_checkpoint(tmp_path):
    """Comme `tiny_checkpoint` de conftest.py, mais intermediate_size=256
    (multiple de 32) : la bascule q3n route TOUT le GPU en q3n, y compris
    mlp.gate_proj -- `tiny_checkpoint` (I=688) n'est pas multiple de 32 et
    fait echouer `quantize_q3n` avant meme d'atteindre self_attn."""
    torch.manual_seed(20260918)
    H, I, L, NH, NKV, V = 256, 256, 1, 8, 2, 512
    d = tmp_path / "hf"; d.mkdir()
    json.dump({
        "architectures": ["LlamaForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
    }, open(d / "config.json", "w"))
    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .02}
    p = "model.layers.0."
    sd[p + "self_attn.q_proj.weight"] = torch.randn(NH * hd, H, dtype=torch.bfloat16) * .02
    sd[p + "self_attn.k_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
    sd[p + "self_attn.v_proj.weight"] = torch.randn(NKV * hd, H, dtype=torch.bfloat16) * .02
    sd[p + "self_attn.o_proj.weight"] = torch.randn(H, NH * hd, dtype=torch.bfloat16) * .02
    sd[p + "mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
    sd[p + "mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16) * .02
    sd[p + "mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16) * .02
    sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def test_symetrique_zero_fixe_a_128_et_une_echelle_par_ligne():
    torch.manual_seed(20260918)
    w = torch.randn(8, 64) * 0.02
    t = _quantize_int8(w, group_size=64, symmetric=True)
    assert isinstance(t, INT8Tensor)
    assert t.scales.shape == (8, 1)                # une echelle par ligne (ng=1)
    assert torch.all(t.zeros == 128)               # point-zero FIXE, pas calcule
    deq = _dequantize_int8(t, torch.float32)
    err = (deq - w).norm() / w.norm()
    assert err < 0.02


def test_temoin_cassant_le_mode_affine_par_defaut_varie_son_point_zero():
    """Bras casse (REGLES §5) : sans symmetric=True, le meme tenseur produit
    des zero-points QUI VARIENT (affine, min/max) -- si ce test devenait
    vert avec `symmetric=False`, la garde ci-dessus ne testerait rien."""
    torch.manual_seed(20260918)
    w = torch.randn(8, 64) * 0.02
    t = _quantize_int8(w, group_size=64, symmetric=False)
    assert not torch.all(t.zeros == 128)


def test_quantize_dispatcher_transmet_symmetric_au_format_int8():
    from acvram.quant.formats import quantize
    torch.manual_seed(20260918)
    w = torch.randn(4, 32) * 0.02
    t_sym = quantize(w, "int8", group_size=32, symmetric=True)
    t_aff = quantize(w, "int8", group_size=32, symmetric=False)
    assert torch.all(t_sym.zeros == 128)
    assert not torch.all(t_aff.zeros == 128)


def _bascule_q3n(plan):
    """Reproduit exactement la bascule de test_improvements.py::
    test_la_bascule_q3n_change_les_couches_pas_seulement_les_etages -- q3n
    est le seul format cible qui force deterministicallement
    self_attn.{q,k,v,o}_proj en int8 (voir TensorRouter.format_for,
    convert.py:358-367), sans dependre du plan de tiering materiel."""
    gpus = {t.name for t in plan.tiers if t.kind == "gpu"}
    for t in plan.tiers:
        if t.kind == "gpu":
            t.weight_format = "q3n"
    for lp in plan.layers:
        if getattr(lp, "fmt", None) and lp.exec_device in gpus | {"cpu"}:
            lp.fmt = "q3n"


def test_attn_qkvo_int8_canal_change_group_size_et_symmetrique_dans_le_manifeste(
        target_rig, tmp_path):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    tiny_checkpoint = _petit_checkpoint(tmp_path)
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    _bascule_q3n(plan)
    out = str(tmp_path / "canal")
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out, attn_qkvo_int8_canal=True),
                       spec=spec)
    manifest = json.load(open(out + "/acvram_manifest.json"))
    assert manifest["attn_int8"] == "canal"
    q = manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]
    assert q["format"] == "int8"
    assert q["symmetrique"] is True
    assert q["group_size"] == q["shape"][1]        # une seule "colonne" = par canal
    # un tenseur int8 HORS q/k/v/o (la tete, plancher int8 sous q3n) garde le
    # groupe de 128 affine -- la bascule est bien scopee aux quatre projections
    head = manifest["tensors"]["lm_head.weight"]
    if head["format"] == "int8":
        assert "symmetrique" not in head
        assert head["group_size"] == 128


def test_promotion_nvfp4_vers_int8_par_plancher_snr_est_aussi_symetrique_par_canal(
        target_rig, tmp_path):
    """18/09, trouve en CONVERTISSANT LE VRAI Qwen3-Coder-30B-A3B : q/k/v/o y
    partent en nvfp4 (20,5-20,7 dB, sous tout plancher raisonnable) et
    n'ATTEIGNENT l'int8 QUE PAR PROMOTION (`PROMOTE["nvfp4"] = "int8"`), pas
    par un routage direct comme sous q3n. Le premier code n'appliquait
    `attn_canal`/`group_size_tenseur` qu'au tenseur nvfp4 initial (jamais
    "int8" a ce point) -- le converti reel sortait en groupe de 128 affine,
    malgre `attn_int8: "canal"` au sommet du manifeste. Reproduit ici sans
    la carte, plancher SNR volontairement haut pour forcer la promotion."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    tiny_checkpoint = _petit_checkpoint(tmp_path)
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = str(tmp_path / "promu")
    convert_checkpoint(tiny_checkpoint, plan,
                       ConversionOptions(out_dir=out, attn_qkvo_int8_canal=True,
                                         snr_floor=100.0, max_promotions=1.0),
                       spec=spec)
    manifest = json.load(open(out + "/acvram_manifest.json"))
    q = manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]
    assert q["format"] == "int8"
    assert q["promoted_from"] == "nvfp4"
    assert q["symmetrique"] is True
    assert q["group_size"] == q["shape"][1]


def test_temoin_cassant_sans_loption_group_size_reste_128(
        target_rig, tmp_path):
    """Bras casse : meme bascule q3n, meme modele, mais sans
    attn_qkvo_int8_canal -- q_proj doit rester au groupe de 128 par defaut,
    sinon le test precedent ne verifierait pas l'option mais un comportement
    deja present."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    tiny_checkpoint = _petit_checkpoint(tmp_path)
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    _bascule_q3n(plan)
    out = str(tmp_path / "groupe")
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=out), spec=spec)
    manifest = json.load(open(out + "/acvram_manifest.json"))
    assert manifest["attn_int8"] == "groupe"
    q = manifest["tensors"]["model.layers.0.self_attn.q_proj.weight"]
    assert q["group_size"] == 128
    assert "symmetrique" not in q
