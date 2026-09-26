"""Pièce 139 (revue/poste5-piece139-mixed-precision-24-09.md) : compressed-tensors « mixed-precision » à dispatch
PAR GROUPE, en opt-in (ACVRAM_HFQUANT_PAR_GROUPE=1). Source jouet dans la disposition d'unsloth/Qwen3.8-27B-NVFP4 :
MLP nvfp4-pack-quantized (groupe 1) ; attention, lm_head et MLP de la DERNIÈRE couche en fp8 par canal (groupe 0,
échelle bf16 [out, 1]) — la dernière couche correspond aux deux motifs, comme les couches 56-63 du vrai ; k_scale /
v_scale du cache fp8 en plus. Chaque test casse si l'on retire la pièce qu'il garde."""
import json
import os

import pytest
import torch

from acvram.quant.hfquant import VAR_PAR_GROUPE, HFQuantCheckpoint, is_hfquant
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4

N_COUCHES = 4          # mini-llama du conftest
DERNIERE = N_COUCHES - 1


def _source_mixte(tmp_path, tiny_checkpoint, variante: str = "") -> str:
    from safetensors.torch import load_file, save_file
    src = load_file(os.path.join(tiny_checkpoint, "model.safetensors"))
    d = tmp_path / f"mixte{variante}"
    d.mkdir()
    cfg = json.load(open(os.path.join(tiny_checkpoint, "config.json")))
    sd = {}
    for k, v in src.items():
        if not (v.dim() == 2 and (".layers." in k or k == "lm_head.weight") and "norm" not in k):
            sd[k] = v
            continue
        base = k[:-len(".weight")]
        mlp = ".mlp." in k
        fp8 = not mlp or f".layers.{DERNIERE}." in k
        if variante == "packe-sur-fp8" and k.endswith("layers.0.self_attn.q_proj.weight"):
            fp8 = False          # visé par le seul groupe fp8, mais empaqueté sur disque
        if variante == "clair-vise" and k == "lm_head.weight":
            sd[k] = v            # visé par le groupe fp8, mais en clair sur disque
            continue
        if fp8:
            s = (v.float().abs().amax(1, keepdim=True) / 448.0).to(torch.bfloat16)
            sd[base + ".weight"] = (v.float() / s.float()).to(torch.float8_e4m3fn)
            sd[base + ".weight_scale"] = s
        else:
            gs = (v.float().abs().amax() / (6 * 448) * 1.37).reshape(())
            q = quantize_nvfp4(v.float(), global_scale=gs)
            sd[base + ".weight_packed"] = q.qweight
            sd[base + ".weight_scale"] = q.block_scale
            sd[base + ".weight_global_scale"] = (torch.ones(1) / q.global_scale).reshape(1)
            sd[base + ".input_global_scale"] = torch.ones(1)
        if k.endswith("self_attn.k_proj.weight"):
            sd[base[:-len(".k_proj")] + ".k_scale"] = torch.tensor(0.5)
            sd[base[:-len(".k_proj")] + ".v_scale"] = torch.tensor(0.5)
    save_file(sd, str(d / "model.safetensors"))
    derniere = f"re:.*layers\\.({DERNIERE})\\.mlp\\.(gate|up|down)_proj$"
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "mixed-precision",
        "config_groups": {
            "group_0": {"format": "float-quantized",
                        "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True},
                        "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$", "re:.*lm_head", derniere]},
            "group_1": {"format": "nvfp4-pack-quantized",
                        "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16},
                        "targets": ["re:.*mlp\\.(gate|up|down)_proj$"]}},
        "ignore": ["re:.*visual.*"]}
    json.dump(cfg, open(d / "config.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(tiny_checkpoint, f)):
            os.symlink(os.path.join(tiny_checkpoint, f), d / f)
    return str(d)


def test_sans_opt_in_le_refus_nomme_reste(tmp_path, tiny_checkpoint, monkeypatch):
    monkeypatch.delenv(VAR_PAR_GROUPE, raising=False)
    path = _source_mixte(tmp_path, tiny_checkpoint)
    assert is_hfquant(path)
    with pytest.raises(NotImplementedError, match="mixed-precision.*non géré"):
        HFQuantCheckpoint(path)


def test_dispatch_par_groupe_nvfp4_direct_et_fp8_canal_exact(tmp_path, tiny_checkpoint, monkeypatch):
    from safetensors.torch import load_file
    monkeypatch.setenv(VAR_PAR_GROUPE, "1")
    path = _source_mixte(tmp_path, tiny_checkpoint)
    src = load_file(os.path.join(path, "model.safetensors"))
    ck = HFQuantCheckpoint(path)
    sortie = dict(ck.iter_tensors(direct_nvfp4=True))
    assert not any(n.endswith(("k_scale", "v_scale", "input_global_scale", "weight_scale", "weight_global_scale"))
                   for n in sortie), sorted(sortie)
    directs = {n: t for n, t in sortie.items() if isinstance(t, NVFP4Tensor)}
    assert len(directs) == 3 * DERNIERE and all(".mlp." in n for n in directs)
    for n, t in directs.items():          # les octets de la source, sans requantification
        assert torch.equal(t.qweight, src[n[:-len(".weight")] + ".weight_packed"])
    fp8 = {n: t for n, t in sortie.items()
           if not isinstance(t, NVFP4Tensor) and t.dtype == torch.float32 and t.dim() == 2}
    assert len(fp8) == 4 * N_COUCHES + 3 + 1, sorted(fp8)          # attention, MLP de la dernière, lm_head
    assert all(f".layers.{DERNIERE}.mlp." in n for n in fp8 if ".mlp." in n)
    for n, t in fp8.items():              # fp32 EXACT : ni arrondi bf16, ni échelle perdue
        base = n[:-len(".weight")]
        assert torch.equal(t, src[n].float() * src[base + ".weight_scale"].float()), n
    # chemin déquantifié : nvfp4 égal au bit au passage direct déquantifié
    deqs = dict(ck.iter_tensors(direct_nvfp4=False))
    for n, t in directs.items():
        assert torch.equal(dequantize_nvfp4(t, torch.bfloat16), deqs[n]), n


@pytest.mark.parametrize("variante,motif", [("packe-sur-fp8", "preuve 'pack'"),
                                            ("clair-vise", "en clair sur disque")])
def test_preuve_discordante_refus_nomme(tmp_path, tiny_checkpoint, monkeypatch, variante, motif):
    monkeypatch.setenv(VAR_PAR_GROUPE, "1")
    path = _source_mixte(tmp_path, tiny_checkpoint, variante)
    with pytest.raises(NotImplementedError, match=motif):
        list(HFQuantCheckpoint(path).iter_tensors(direct_nvfp4=True))


def test_convert_fp8_en_int8_par_canal_jamais_bf16_clair(tmp_path, tiny_checkpoint, target_rig, monkeypatch):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    monkeypatch.setenv(VAR_PAR_GROUPE, "1")
    path = _source_mixte(tmp_path, tiny_checkpoint)
    spec = load_model_spec(path, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = tmp_path / "converti"
    convert_checkpoint(path, plan, ConversionOptions(out_dir=str(out), awq=False, passage_direct=True), spec=spec)
    t = json.load(open(out / "acvram_manifest.json"))["tensors"]
    fp8 = {k: v for k, v in t.items() if v.get("origine") == "fp8"}
    assert len(fp8) == 4 * N_COUCHES + 3 + 1, sorted(fp8)
    for k, v in fp8.items():
        assert v["format"] == "int8" and v.get("passage_direct") != "clair", (k, v)
        assert v["group_size"] == v["shape"][1], (k, v)             # une échelle par ligne de sortie
    assert sum(1 for v in t.values() if v.get("passage_direct") is True) == 3 * DERNIERE
    from acvram.engine.loader import load_model
    from tests.test_engine import _prefill
    charge = load_model(str(out), dtype=torch.float32, device_override="cpu")
    assert torch.isfinite(_prefill(charge.model, [5, 42, 7, 99, 13])).all()
    # bout en bout : TOUT poids int8 servi (piles gate_up et vues comprises) porte la marque préfill bf16, sinon le
    # premier préfill sur carte reconstruit la copie signée (OOM des 24/09 08:26 et 09:1x)
    from acvram.engine.layers import QuantLinear
    from acvram.kernels import _i8c_poids
    from acvram.quant.formats import INT8Tensor
    i8 = [(n, m.qweight) for n, m in charge.model.named_modules()
          if isinstance(m, QuantLinear) and isinstance(m.qweight, INT8Tensor)]
    assert i8 and [n for n, q in i8 if not q.__dict__.get("prefill_bf16")] == []
    assert all(_i8c_poids(q) is None for _, q in i8)


def _source_modelopt(tmp_path, algo: str, fp8: bool) -> str:
    from safetensors.torch import save_file
    d = tmp_path / f"modelopt-{algo}-{int(fp8)}"
    d.mkdir()
    w = torch.randn(32, 32)
    sd = {"model.layers.0.self_attn.q_proj.weight": w.to(torch.float8_e4m3fn) if fp8 else w.to(torch.bfloat16),
          "model.layers.0.self_attn.q_proj.weight_scale": torch.tensor(0.01)}
    save_file(sd, str(d / "model.safetensors"))
    json.dump({"quantization_config": {"quant_method": "modelopt", "quant_algo": algo}}, open(d / "config.json", "w"))
    return str(d)


def test_modelopt_mixed_precision_refus_nomme(tmp_path):
    path = _source_modelopt(tmp_path, "MIXED_PRECISION", fp8=True)
    assert is_hfquant(path)
    with pytest.raises(NotImplementedError, match="MIXED_PRECISION non géré"):
        HFQuantCheckpoint(path)


def test_modelopt_fp8_brut_refus_nomme(tmp_path):
    # hors MIXED_PRECISION, un poids F8 sans weight_scale_2 ne sort jamais brut (échelle perdue)
    path = _source_modelopt(tmp_path, "FP8", fp8=True)
    with pytest.raises(NotImplementedError, match="FP8 modelopt sans weight_scale_2"):
        list(HFQuantCheckpoint(path).iter_tensors())
    assert dict(HFQuantCheckpoint(_source_modelopt(tmp_path, "FP8", fp8=False)).iter_tensors())


def test_sans_vision_alias_texte_au_bit_de_la_conversion_vl(tmp_path, target_rig):
    """--sans-vision (ordre chef) : manifeste vision=non, aucun tenseur de la tour, et tenseurs texte identiques
    AU BIT à ceux de la conversion qui garde la tour."""
    from tests.test_mm_conversion import VISION_PREFIXES as PREF, _ecrire_source, _empreintes
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    src = _ecrire_source(tmp_path / "src_vl", vision=True)
    spec = load_model_spec(src)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    avec, sans = str(tmp_path / "avec"), str(tmp_path / "sans")
    convert_checkpoint(src, plan, ConversionOptions(out_dir=avec), spec=spec)
    convert_checkpoint(src, plan, ConversionOptions(out_dir=sans, sans_vision=True), spec=spec)
    m = json.load(open(os.path.join(sans, "acvram_manifest.json")))
    assert m["vision"] == "non" and m["vision_bytes"] == 0 and m["vision_ecartee"] > 0
    assert not [k for k in m["tensors"] if k.startswith(PREF)]
    e_sans = _empreintes(sans)
    assert e_sans and not [k for k in e_sans if k.startswith(PREF)]
    assert e_sans == {k: v for k, v in _empreintes(avec).items() if not k.startswith(PREF)}


def _i8c(n=64, k=256):
    from acvram.quant.formats import _quantize_int8
    torch.manual_seed(139)
    return _quantize_int8(torch.randn(n, k), k, symmetric=True)


def test_poids_origine_fp8_sans_copie_signee_au_prefill():
    """Casse si la copie int8 signée du chemin cublas revient sur un poids marqué (OOM du 24/09 : +10,6 Go)."""
    from acvram.kernels import _i8c_poids
    temoin = _i8c()
    assert _i8c_poids(temoin) is not None                    # le chemin qkvo garde sa copie, inchangé
    t = _i8c()
    t.__dict__["prefill_bf16"] = True
    assert _i8c_poids(t) is None and t.__dict__["_i8c"] is False


def test_vue_g128_dequant_au_bit_du_par_canal():
    from acvram.kernels import vue_g128
    from acvram.quant.formats import _dequantize_int8
    t = _i8c()
    v = vue_g128(t)
    assert v.group_size == 128 and v.qweight is t.qweight
    assert torch.equal(_dequantize_int8(v, torch.float32), _dequantize_int8(t, torch.float32))


def test_chargeur_marque_origine_fp8_et_regime_le_nomme():
    from acvram import regime
    from acvram.engine.loader import _build_quant
    t = _i8c()
    sd = {"w.qweight": t.qweight, "w.scales": t.scales, "w.zeros": t.zeros}
    lecteur = type("L", (), {"get": staticmethod(lambda k: sd[k])})()
    base = {"format": "int8", "shape": list(t.shape), "keys": list(sd), "group_size": t.group_size}
    assert _build_quant({**base, "origine": "fp8"}, "w", lecteur, 128).__dict__.get("prefill_bf16") is True
    assert not _build_quant(base, "w", lecteur, 128).__dict__.get("prefill_bf16")
    try:
        regime.declarer_modele_charge({"vision": "non", "tensors": {"a": {**base, "origine": "fp8"},
                                                                     "b": {**base, "origine": "fp8"}, "c": base}})
        assert regime.prefill_i8c_texte() == "prefill_int8=bf16(origine fp8 ×2)"
        assert "prefill_int8=bf16(origine fp8 ×2)" in regime.regime_ligne()
    finally:
        regime.declarer_modele_charge(None)
    assert regime.prefill_i8c_texte() is None


def test_marque_prefill_bf16_suit_la_pile_int8():
    """Casse si la pile gate_up (stack_int8_linears) perd la marque : la copie signée revenait (OOM 08:26)."""
    from acvram.engine.layers import QuantLinear, stack_int8_linears
    from acvram.kernels import _i8c_poids
    lins = []
    for _ in range(2):
        t = _i8c()
        t.__dict__["prefill_bf16"] = True
        lins.append(QuantLinear(t))
    pile = stack_int8_linears(lins)
    assert pile is not None and pile.qweight.__dict__.get("prefill_bf16") is True
    assert all(l.qweight.__dict__.get("prefill_bf16") is True for l in lins)
    assert _i8c_poids(pile.qweight) is None
    temoin = stack_int8_linears([QuantLinear(_i8c()), QuantLinear(_i8c())])
    assert not temoin.qweight.__dict__.get("prefill_bf16")


def test_marque_prefill_bf16_survit_au_deplacement():
    """Casse si INT8Tensor.to perd la marque (le chargeur déplace les poids sur la carte après _build_quant)."""
    t = _i8c()
    t.__dict__["prefill_bf16"] = True
    assert t.to("cpu").__dict__.get("prefill_bf16") is True
    assert not _i8c().to("cpu").__dict__.get("prefill_bf16")
