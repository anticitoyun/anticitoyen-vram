"""Pièce 32 : compteur d observations par expert pendant la calibration et
refus nommé quand le corpus ne les atteint pas (25(a) : l échelle AWQ n est
stable qu à ≥ 512 observations). La conversion ne change RIEN quand le
critère est tenu (sortie au bit) ; le rapport va au manifeste."""
import json
import os

import pytest
import torch

from acvram.quant.calibrate import ActStats
from acvram.quant.collect import observations_par_expert, rapport_observations


def _stats(par):
    """par = {(couche, expert) : n} → statistiques des trois projections."""
    out = {}
    for (c, e), n in par.items():
        for proj in ("gate_proj", "up_proj", "down_proj"):
            out[f"model.layers.{c}.mlp.experts.{e}.{proj}.weight"] = ActStats(
                mean_abs=torch.ones(8), max_abs=None, n_samples=n)
    out["model.layers.0.self_attn.q_proj.weight"] = ActStats(mean_abs=torch.ones(8), max_abs=None, n_samples=9999)
    return out


def test_observations_par_expert_prend_le_minimum():
    s = _stats({(0, 1): 700, (0, 2): 40, (1, 1): 0})
    s["model.layers.0.mlp.experts.1.down_proj.weight"] = ActStats(mean_abs=torch.ones(8), max_abs=None, n_samples=300)
    assert observations_par_expert(s) == {(0, 1): 300, (0, 2): 40, (1, 1): 0}     # minimum sur les projections
    assert observations_par_expert({}) == {}


def test_rapport_et_refus():
    r = rapport_observations(_stats({(0, 1): 700, (0, 2): 40, (1, 1): 0, (1, 2): 900}), 512)
    assert r["experts"] == 4 and r["minimum"] == 0 and r["maximum"] == 900
    assert r["sous_seuil"] == 2 and r["jamais_routes"] == 1 and r["part_sous_seuil"] == 0.5
    assert r["suffisant"] is False and r["facteur_corpus_pour_atteindre"] is None   # minimum 0 : aucun facteur ne suffit
    assert set(r["exemples_sous_seuil"]) == {"0/2", "1/1"}
    r2 = rapport_observations(_stats({(0, 1): 700, (0, 2): 256}), 512)
    assert r2["suffisant"] is False and r2["facteur_corpus_pour_atteindre"] == 2.0   # ×2 de corpus
    r3 = rapport_observations(_stats({(0, 1): 700, (0, 2): 512}), 512)
    assert r3["suffisant"] is True and r3["sous_seuil"] == 0
    assert rapport_observations({}, 512)["suffisant"] is None                        # dense : pas d avis


@pytest.mark.parametrize("obs", [None, {"experts": 2, "minimum": 700, "suffisant": True, "obs_min_demande": 512}])
def test_conversion_inchangee_et_rapport_au_manifeste(tiny_checkpoint, target_rig, tmp_path, obs):
    """Sortie AU BIT avec et sans rapport ; le rapport va au manifeste."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=64, max_concurrent_seqs=1))
    stats = {"__observations__": obs} if obs else {}
    out = str(tmp_path / f"out-{'avec' if obs else 'sans'}")
    convert_checkpoint(tiny_checkpoint, plan, ConversionOptions(out_dir=out, quant_device="cpu"),
                       spec=spec, stats=dict(stats))
    m = json.load(open(os.path.join(out, "acvram_manifest.json")))
    assert ("observations_experts" in m) == bool(obs)
    if obs:
        assert m["observations_experts"]["minimum"] == 700
    # les octets des poids sont les mêmes dans les deux conversions (au bit)
    import hashlib
    shards = sorted(f for f in os.listdir(out) if f.endswith(".safetensors"))
    assert shards, os.listdir(out)
    sha = hashlib.sha256(b"".join(open(os.path.join(out, f), "rb").read() for f in shards)).hexdigest()
    ref = tmp_path.parent / "sha-obs.txt"
    if ref.exists():
        assert sha == ref.read_text(), "la sortie change selon la présence du rapport"
    else:
        ref.write_text(sha)
