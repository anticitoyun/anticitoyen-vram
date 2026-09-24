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
