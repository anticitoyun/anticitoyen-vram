"""11e (poste1, 01/10, à sec avant la prise) : `convert --echelle balayage-w` avec AWQ (le défaut, celui des quatre convertis
du scellé) n'a jamais tourné. Les tests de la règle passent `use_awq=False` (test_echelle_cli_11e.py). La grille AWQ
(`search_channel_scales` → `_quant_dequant`, calibrate.py:260-263) quantifie sous la règle GLOBALE balayage-w sans
transmettre d'importance → ValueError (nvfp4.py:380) au premier tenseur calibré. La prise de conversion aurait été perdue.
Ce cliquet convertit le petit checkpoint avec de VRAIES statistiques (collect_activation_stats) : rouge tant que la recherche
AWQ ne sait pas tourner sous balayage-w. Le correctif est une décision de méthode (chef / autrice), pas de ce test."""
import json
import os

import pytest
import torch

from acvram.quant import nvfp4


@pytest.fixture(autouse=True)
def _max6_ensuite():
    yield
    nvfp4.regler_echelle("max6")


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="11e : la grille AWQ quantifie sous balayage-w sans importance (calibrate.py:260-263)")
def test_conversion_balayage_w_avec_awq_et_vraies_statistiques(tiny_checkpoint, target_rig, tmp_path):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.collect import collect_activation_stats
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=256, max_concurrent_seqs=2, force_format="nvfp4"))
    g = torch.Generator().manual_seed(0)
    calib = [torch.randint(0, spec.vocab_size, (64,), generator=g).tolist() for _ in range(4)]
    stats = collect_activation_stats(tiny_checkpoint, spec, calib, device="cpu")
    stats.pop("__observations__", None)
    assert stats, "aucune statistique : le test ne jugerait pas le chemin AWQ"
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=str(tmp_path), echelle_nvfp4="balayage-w"),
                       spec=spec, stats=stats)
    m = json.load(open(os.path.join(tmp_path, "acvram_manifest.json")))
    assert m["echelle_nvfp4"]["regle"] == "balayage-w" and m["echelle_nvfp4"]["part_balayes"] > 0
