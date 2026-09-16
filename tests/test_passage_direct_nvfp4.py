"""Passage DIRECT d'une source déjà NVFP4 (poste7, revue/poste7-convertisseur-formats-16-09.md
§ 3.1) : modelopt (``weight`` u8, ``weight_scale`` e4m3, ``weight_scale_2`` f32)
et compressed-tensors nvfp4-pack-quantized (``weight_packed``, ``weight_scale``,
``weight_global_scale``) deviennent des ``NVFP4Tensor`` sans déquantifier ni
requantifier — les poids servis sont ceux de vLLM. Contrats : la
déquantification du passage direct est égale AU BIT à celle du lecteur
`hfquant` (chemin déquantifié) ; les octets écrits par `acvram convert
--passage-direct` sont ceux de la source ; le converti se charge et prédit ;
et le doute § 4 de poste7 est tranché par la mesure : requantifier ne
reproduit pas les codes de la source (mesuré 0/3 sur GLM et Coder réels ;
ici sur la source jouet)."""
import json
import os

import pytest
import torch

from acvram.quant.hfquant import HFQuantCheckpoint, is_hfquant
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4


def _source_nvfp4(tmp_path, tiny_checkpoint, methode: str) -> str:
    """Le mini-modèle llama du conftest, requantifié en NVFP4 et écrit dans la
    disposition de ``methode`` (modelopt | compressed-tensors) ; lm_head et
    les normes restent en clair, comme le font ces producteurs."""
    from safetensors.torch import load_file, save_file
    src = load_file(os.path.join(tiny_checkpoint, "model.safetensors"))
    d = tmp_path / methode
    d.mkdir()
    cfg = json.load(open(os.path.join(tiny_checkpoint, "config.json")))
    sd = {}
    torch.manual_seed(3)
    for k, v in src.items():
        quantifiable = v.dim() == 2 and ".layers." in k and "norm" not in k
        if not quantifiable:
            sd[k] = v
            continue
        # échelle globale VOLONTAIREMENT différente de amax/(6·448) : les
        # codes ne peuvent pas être retrouvés par une requantification
        gs = (v.float().abs().amax() / (6 * 448) * 1.37).reshape(())
        q = quantize_nvfp4(v.float(), global_scale=gs)
        base = k[:-len(".weight")]
        if methode == "modelopt":
            sd[base + ".weight"] = q.qweight
            sd[base + ".weight_scale"] = q.block_scale
            sd[base + ".weight_scale_2"] = q.global_scale.reshape(())
            sd[base + ".input_scale"] = torch.tensor(1.0)
        else:
            sd[base + ".weight_packed"] = q.qweight
            sd[base + ".weight_scale"] = q.block_scale
            sd[base + ".weight_global_scale"] = (torch.ones(()) / q.global_scale).reshape(())
    save_file(sd, str(d / "model.safetensors"))
    if methode == "modelopt":
        cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                      "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 16}}}}
        json.dump({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4", "group_size": 16}},
                  open(d / "hf_quant_config.json", "w"))
    else:
        cfg["quantization_config"] = {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                      "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float",
                                                                                "strategy": "tensor_group",
                                                                                "group_size": 16, "symmetric": True}}}}
    json.dump(cfg, open(d / "config.json", "w"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(tiny_checkpoint, f)):
            os.symlink(os.path.join(tiny_checkpoint, f), d / f)
    return str(d)


@pytest.mark.parametrize("methode", ["modelopt", "compressed-tensors"])
def test_direct_egal_hfquant_au_bit_et_pas_la_requantification(tmp_path, tiny_checkpoint, methode):
    path = _source_nvfp4(tmp_path, tiny_checkpoint, methode)
    assert is_hfquant(path)
    ck = HFQuantCheckpoint(path)
    directs = {n: t for n, t in ck.iter_tensors(direct_nvfp4=True) if isinstance(t, NVFP4Tensor)}
    assert len(directs) == 4 * 7, len(directs)
    deqs = dict(ck.iter_tensors(direct_nvfp4=False))
    for n, t in directs.items():
        assert torch.equal(dequantize_nvfp4(t, torch.bfloat16), deqs[n]), f"{n} : direct ≠ hfquant"
        assert t.shape == tuple(deqs[n].shape) and t.padded_in == t.shape[1]
        # doute § 4 : requantifier la déquantifiée ne rend pas les codes de la source
        rq = quantize_nvfp4(deqs[n].float())
        assert not torch.equal(rq.qweight, t.qweight), f"{n} : la requantification reproduirait la source"
    # les couches en clair sortent inchangées, quel que soit le mode
    clairs = {n: t for n, t in ck.iter_tensors(direct_nvfp4=True) if not isinstance(t, NVFP4Tensor)}
    assert "lm_head.weight" in clairs and clairs["lm_head.weight"].dtype == torch.bfloat16


def test_convert_passage_direct_ecrit_les_octets_de_la_source(tmp_path, tiny_checkpoint, target_rig):
    from safetensors.torch import load_file
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    path = _source_nvfp4(tmp_path, tiny_checkpoint, "modelopt")
    spec = load_model_spec(path, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    out = tmp_path / "direct"
    convert_checkpoint(path, plan, ConversionOptions(out_dir=str(out), awq=False, passage_direct=True), spec=spec)
    manifeste = json.load(open(out / "acvram_manifest.json"))
    entrees = {k: v for k, v in manifeste["tensors"].items() if v.get("passage_direct") is True}
    assert len(entrees) == 4 * 7 and all(v["format"] == "nvfp4" for v in entrees.values())
    # les couches gardées en clair par la source (lm_head ici, comme l'ignore de
    # vLLM) restent en clair, quel que soit le plan (poste3, 17/09)
    tete = manifeste["tensors"]["lm_head.weight"]
    assert tete["format"] == "bf16" and tete.get("passage_direct") == "clair", tete
    src = load_file(os.path.join(path, "model.safetensors"))
    ecrit = {}
    for f in sorted(os.listdir(out)):
        if f.endswith(".safetensors"):
            ecrit.update(load_file(str(out / f)))
    for n in entrees:
        base = n[:-len(".weight")]
        assert torch.equal(ecrit[n + ".qweight"], src[base + ".weight"])
        assert torch.equal(ecrit[n + ".block_scale"], src[base + ".weight_scale"].view(torch.uint8))
        assert torch.equal(ecrit[n + ".global_scale"].reshape(()), src[base + ".weight_scale_2"].reshape(()))
    # le converti se charge et prédit (processeur)
    from acvram.engine.loader import load_model
    from tests.test_engine import _prefill
    loaded = load_model(str(out), dtype=torch.float32, device_override="cpu")
    logits = _prefill(loaded.model, [5, 42, 7, 99, 13])
    assert torch.isfinite(logits).all()
