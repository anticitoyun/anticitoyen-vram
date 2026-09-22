"""fp8 statique (Devstral-Small-2, jerome 18/09) : quant_method="fp8",
weight_block_size=null -- pas de blocs, une echelle SCALAIRE par tenseur.
config.json REEL verifie (huggingface.co/mistralai/Devstral-Small-2-24B-
Instruct-2512, 18/09) : {"activation_scheme": "static", "dequantize": false,
"modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector",
"lm_head"], "quant_method": "fp8", "weight_block_size": null}."""
import json
import os

import torch
from safetensors.torch import load_file, save_file

from acvram.quant.hfquant import HFQuantCheckpoint, is_hfquant


def _source_fp8(tmp_path, tiny_checkpoint) -> str:
    """Le mini-modele llama du conftest, requantifie en fp8 statique par
    tenseur ; lm_head reste en clair, comme `modules_to_not_convert`."""
    src = load_file(os.path.join(tiny_checkpoint, "model.safetensors"))
    d = tmp_path / "fp8"
    d.mkdir()
    cfg = json.load(open(os.path.join(tiny_checkpoint, "config.json")))
    sd = {}
    for k, v in src.items():
        quantifiable = v.dim() == 2 and ".layers." in k and "norm" not in k
        if not quantifiable or k.startswith("lm_head"):
            sd[k] = v
            continue
        scale = (v.float().abs().amax() / 448.0).clamp(min=1e-12).reshape(())
        q = (v.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        base = k[:-len(".weight")]
        sd[base + ".weight"] = q
        sd[base + ".weight_scale"] = scale
        sd[base + ".input_scale"] = torch.tensor(1.0)
    save_file(sd, str(d / "model.safetensors"))
    cfg["quantization_config"] = {
        "activation_scheme": "static", "dequantize": False,
        "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"],
        "quant_method": "fp8", "weight_block_size": None,
    }
    json.dump(cfg, open(d / "config.json", "w"))
    return str(d)


def test_is_hfquant_reconnait_fp8(tmp_path, tiny_checkpoint):
    assert is_hfquant(_source_fp8(tmp_path, tiny_checkpoint))


def test_dequantification_fp8_par_tenseur_proche_de_la_source(tmp_path, tiny_checkpoint):
    d = _source_fp8(tmp_path, tiny_checkpoint)
    src = load_file(os.path.join(tiny_checkpoint, "model.safetensors"))
    ckpt = HFQuantCheckpoint(d)
    dequant = dict(ckpt.iter_tensors())
    nom = "model.layers.0.self_attn.q_proj.weight"
    err = (dequant[nom].float() - src[nom].float()).norm() / src[nom].float().norm()
    assert err < 0.05, f"erreur fp8 trop grande : {err}"
    assert dequant[nom].dtype == torch.bfloat16


def test_lm_head_hors_modules_a_convertir_reste_inchange(tmp_path, tiny_checkpoint):
    """`modules_to_not_convert` (lm_head ici) n'a pas de `.weight_scale`
    compagnon -- doit ressortir tel quel, dtype d'origine, pas dequantifie."""
    d = _source_fp8(tmp_path, tiny_checkpoint)
    src = load_file(os.path.join(tiny_checkpoint, "model.safetensors"))
    dequant = dict(HFQuantCheckpoint(d).iter_tensors())
    assert torch.equal(dequant["lm_head.weight"], src["lm_head.weight"])


def test_temoin_cassant_quant_method_autre_nest_pas_fp8(tmp_path, tiny_checkpoint):
    d = _source_fp8(tmp_path, tiny_checkpoint)
    cfg_path = os.path.join(d, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"]["quant_method"] = "autre_chose"
    json.dump(cfg, open(cfg_path, "w"))
    assert not is_hfquant(d)
