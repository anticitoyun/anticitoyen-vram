"""Le lecteur EXL3 : détection et garde-fous.

La reconstruction elle-même est déléguée à exllamav3 (voir quant/exl3.py) et
vérifiée de bout en bout par la conversion réelle d'un modèle — un treillis mal
lu produit du charabia immédiat, pas une erreur discrète.
"""

import json

import pytest
import torch

from acvram.quant.exl3 import EXL3Checkpoint, is_exl3


def test_is_exl3(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    assert not is_exl3(str(d))
    json.dump({"architectures": ["MistralForCausalLM"]},
              open(d / "config.json", "w"))
    assert not is_exl3(str(d))
    json.dump({"architectures": ["MistralForCausalLM"],
               "quantization_config": {"quant_method": "exl3", "bits": 4}},
              open(d / "config.json", "w"))
    assert is_exl3(str(d))


def test_rename_language_model():
    assert EXL3Checkpoint._rename("model.language_model.layers.0.mlp.up_proj") \
        == "model.layers.0.mlp.up_proj"
    assert EXL3Checkpoint._rename("lm_head.weight") == "lm_head.weight"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="la reconstruction EXL3 exige CUDA")
def test_checkpoint_requires_files(tmp_path):
    d = tmp_path / "vide"
    d.mkdir()
    json.dump({"quantization_config": {"quant_method": "exl3"}},
              open(d / "config.json", "w"))
    with pytest.raises(FileNotFoundError):
        EXL3Checkpoint(str(d))
