"""Pièce 131 bis (verdict pièce 131, 24/09) : un point de contrôle compressed-tensors « mixed-precision »
(plusieurs config_groups à formats différents, ex. Qwen3.5-VL unsloth/Qwen3.8-27B-NVFP4 : nvfp4-pack-
quantized + float-quantized) n'a aucun dispatch par tenseur dans HFQuantCheckpoint — sans refus explicite,
la conversion serait partielle (poids manquants) ou pire, fp8 non déquantifié et silencieusement faux.
`HFQuantCheckpoint.__init__` doit refuser NOMMÉMENT, jamais laisser passer. Config.json factice, sans
poids : le refus arrive avant toute lecture de tenseur."""
import json
import os

import pytest

from acvram.quant.hfquant import HFQuantCheckpoint, is_hfquant


def _config_json(tmp_path, quantization_config: dict) -> str:
    d = tmp_path / "source"
    d.mkdir()
    json.dump({"quantization_config": quantization_config}, open(os.path.join(d, "config.json"), "w"))
    return str(d)


def test_mixed_precision_refuse(tmp_path):
    chemin = _config_json(tmp_path, {
        "quant_method": "compressed-tensors", "format": "mixed-precision",
        "config_groups": {
            "group_0": {"format": "float-quantized", "targets": ["re:.*self_attn.*"]},
            "group_1": {"format": "nvfp4-pack-quantized", "targets": ["re:.*mlp.*"]},
        },
    })
    assert is_hfquant(chemin) is True          # reconnu comme hfquant : le refus vient de __init__, pas d'ici
    with pytest.raises(NotImplementedError, match="mixed-precision"):
        HFQuantCheckpoint(chemin)


def test_groupes_formats_differents_sans_mixed_precision_aussi_refuse(tmp_path):
    """Le nom « mixed-precision » n'est pas la seule façon de le dire : des config_groups à formats
    différents, sous un autre format de premier niveau, doivent refuser aussi (défense en profondeur)."""
    chemin = _config_json(tmp_path, {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {"format": "float-quantized", "targets": ["re:.*self_attn.*"]},
            "group_1": {"format": "nvfp4-pack-quantized", "targets": ["re:.*mlp.*"]},
        },
    })
    with pytest.raises(NotImplementedError):
        HFQuantCheckpoint(chemin)


def test_format_unique_ne_refuse_pas(tmp_path):
    """Contrôle négatif : un seul format sur tous les groupes ne déclenche pas le refus — sinon le test
    du dessus passerait pour une mauvaise raison (n'importe quel refus, pas celui de l'hétérogénéité)."""
    chemin = _config_json(tmp_path, {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {"format": "nvfp4-pack-quantized", "weights": {"num_bits": 4},
                       "targets": ["re:.*mlp.*"]},
        },
    })
    HFQuantCheckpoint(chemin)          # ne lève pas
