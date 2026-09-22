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


def test_rapport_separe_les_trois_populations():
    """Pièce 45 : un expert jamais (ou quasi jamais) routé ne compte pas dans
    le facteur de corpus — il ne recevra pas d échelle AWQ de toute façon."""
    r = rapport_observations(_stats({(0, 1): 700, (0, 2): 40, (1, 1): 0, (1, 2): 900}), 512)
    assert r["experts"] == 4 and r["minimum"] == 0 and r["maximum"] == 900
    assert r["jamais_routes"] == 1 and r["quasi_jamais_routes"] == 0
    assert r["atteints"] == 3 and r["p10_atteints"] == 40
    assert r["sous_seuil_parmi_atteints"] == 1
    assert r["suffisant"] is False and r["facteur_corpus_pour_atteindre"] == 12.8     # 512 / 40, pas 512 / 0
    assert set(r["exemples_sous_seuil"]) == {"0/2"} and "1/1" in r["exemples_non_atteints"]
    assert "pièce 27" in r["renvoi"]


def test_le_facteur_ignore_les_quasi_jamais_routes():
    """La faute du 22/09 : 3 522 experts sous 512 tiraient un facteur ×512 et
    refusaient la conversion. Avec le p10 des ATTEINTS, le facteur est celui
    que le corpus peut réellement corriger."""
    par = {(0, e): 600 for e in range(20)}
    par.update({(1, e): 3 for e in range(30)})        # quasi jamais routés : hors calcul
    r = rapport_observations(_stats(par), 512)
    assert r["quasi_jamais_routes"] == 30 and r["atteints"] == 20
    assert r["suffisant"] is True and r["facteur_corpus_pour_atteindre"] <= 1.0
    assert r["part_non_atteints"] == 0.6


def test_jetons_et_minutes_necessaires():
    r = rapport_observations(_stats({(0, 1): 700, (0, 2): 256}), 512,
                             corpus_jetons=16384, secondes=120.0)
    assert r["suffisant"] is False and r["facteur_corpus_pour_atteindre"] == 2.0
    assert r["jetons_pour_p10"] == 32768 and r["minutes_pour_p10"] == 4.0


def test_cas_limites():
    r3 = rapport_observations(_stats({(0, 1): 700, (0, 2): 512}), 512)
    assert r3["suffisant"] is True and r3["sous_seuil_parmi_atteints"] == 0
    assert rapport_observations({}, 512)["suffisant"] is None                        # dense : pas d avis
    aucun = rapport_observations(_stats({(0, 1): 0, (0, 2): 2}), 512)                 # rien d atteint
    assert aucun["suffisant"] is False and aucun["atteints"] == 0 and "routage" in aucun["raison"]


def test_le_seuil_de_routage_suit_le_convertisseur():
    """SEUIL_EXPERT_ROUTE doit rester égal à MIN_ECHANTILLONS_AWQ : sinon le
    rapport compte comme « atteint » un expert que la conversion laisse à
    l identité, et le facteur de corpus ment à nouveau."""
    import pathlib
    import re as _re
    from acvram.quant.collect import SEUIL_EXPERT_ROUTE
    src = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "quant" / "convert.py").read_text()
    m = _re.search(r"MIN_ECHANTILLONS_AWQ = (\d+)", src)
    assert m and int(m.group(1)) == SEUIL_EXPERT_ROUTE


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


def test_corpus_jetons_derive_les_sequences():
    """`--corpus-jetons N` dérive le nombre de séquences de `--calib-len` :
    c est le chiffre que le refus imprime, donc celui qu on doit pouvoir
    recopier tel quel dans la commande suivante."""
    import argparse

    from acvram.cli import build_parser
    p = build_parser()
    a = p.parse_args(["convert", "modele", "--corpus-jetons", "65536", "--calib-len", "512"])
    assert a.corpus_jetons == 65536 and a.calib_len == 512
    assert max(1, -(-a.corpus_jetons // a.calib_len)) == 128          # 128 séquences
    b = p.parse_args(["convert", "modele"])
    assert b.corpus_jetons == 0 and b.calib_seqs == 32                # défaut inchangé : 16 384 jetons
