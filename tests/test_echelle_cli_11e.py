"""11e (poste6, 30/09, ordre chef) : `acvram convert --echelle {max6,4sur6,balayage,balayage-w}` exposé ; défaut max6
INCHANGÉ (le test au bit du défaut vit dans test_nvfp4_4sur6 / test_nvfp4_balayage, gardés). À sec :
(1) le parseur accepte les quatre règles, refuse une cinquième, défaut max6 ; (2) une conversion `balayage` du tiny
checkpoint porte la règle, `part_balayes` > 0 et `sous_normales` au manifeste (total et par tenseur) ; (3) `balayage-w`
sans statistiques de calibration → refus nommé (pas de secours silencieux) ; avec des ActStats → importance (E|x|/s)²
transmise, blocs balayés > 0 ; (4) max6 reste la règle du module après un parseur par défaut.
Cassure : retirer « balayage » des choices → (1) rouge ; retirer `supplement` de quantize_with_calibration → (3) rouge."""
import json
import os

import pytest
import torch

from acvram.cli import build_parser
from acvram.quant import nvfp4
from acvram.quant.calibrate import ActStats, quantize_with_calibration


@pytest.fixture(autouse=True)
def _max6_ensuite():
    yield
    nvfp4.regler_echelle("max6")


def test_parseur_quatre_regles_defaut_max6():
    p = build_parser()
    assert p.parse_args(["convert", "m"]).echelle == "max6"
    for e in ("max6", "4sur6", "balayage", "balayage-w"):
        assert p.parse_args(["convert", "m", "--echelle", e]).echelle == e
    with pytest.raises(SystemExit):
        p.parse_args(["convert", "m", "--echelle", "absmax"])
    assert nvfp4.echelle_courante() == "max6"


def test_conversion_balayage_au_manifeste(tiny_checkpoint, target_rig, tmp_path):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=str(tmp_path), echelle_nvfp4="balayage"), spec=spec)
    m = json.load(open(os.path.join(tmp_path, "acvram_manifest.json")))
    assert m["options"]["echelle_nvfp4"] == "balayage"
    tot = m["echelle_nvfp4"]
    assert tot["regle"] == "balayage" and tot["blocs"] > 0 and 0 < tot["part_balayes"] <= 1 and tot["sous_normales"] >= 0
    par_t = [e["echelle"] for e in m["tensors"].values() if e.get("format") == "nvfp4" and "echelle" in e]
    assert par_t and sum(e["blocs"] for e in par_t) == tot["blocs"]
    assert all("part_balayes" in e and "sous_normales" in e for e in par_t)


def test_balayage_w_sans_stats_replie_en_balayage_et_pese_par_les_stats(capsys):
    torch.manual_seed(3)
    w = torch.randn(64, 256) * (torch.rand(64, 1) + 0.1)
    nvfp4.regler_echelle("balayage-w")
    # 11e option A (01/10) : un tenseur sans statistiques (tête, hors calibration) passe en balayage (MSE), dit une fois —
    # lever ici (version du 30/09) perdait toute la conversion AWQ au premier tenseur sans stats (B1, test_balayage_w_awq_11e)
    import acvram.quant.calibrate as C
    C._SANS_STATS_DIT = False
    qt0, _, _ = quantize_with_calibration(w, "nvfp4", None, use_awq=False)
    assert qt0.echelle_stats["echelle"] == "balayage" and "balayage (MSE)" in capsys.readouterr().out
    stats = ActStats((torch.rand(256) * 3 + 0.01), None, 8)
    qt, scaler, _ = quantize_with_calibration(w, "nvfp4", stats, use_awq=False)
    assert qt.echelle_stats["echelle"] == "balayage-w" and qt.echelle_stats["balayes"] > 0
    # même poids, même stats, règle balayage (MSE) : les échelles diffèrent (la pondération a agi)
    nvfp4.regler_echelle("balayage")
    qt2, _, _ = quantize_with_calibration(w, "nvfp4", stats, use_awq=False)
    assert not torch.equal(qt.block_scale.view(torch.uint8), qt2.block_scale.view(torch.uint8))
