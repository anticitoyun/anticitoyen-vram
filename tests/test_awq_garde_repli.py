"""poste7-awq-relu2-garde-repli-17-09 : geste (b). La garde de norme
(commit 2a68aa6) refusait TOUTE la conversion dès qu'un seul tenseur
sortait de [0,80;1,25] -- exercé en vrai sur Nemotron calibA le 17/09 :
145/346 tenseurs `mlp.experts.*.down_proj` hors bornes malgré le seuil de
8 échantillons (`MIN_ECHANTILLONS_AWQ`), cause réelle identifiée
ailleurs (ReLU² creux, pas le nombre d'échantillons). Le refus total
laissait aucune sortie utilisable pour un défaut LOCALISÉ. Ce fichier
teste le repli PAR TENSEUR : identité pour le tenseur fautif, conversion
qui aboutit, sous un plafond de 50 % (au-delà, plus une réparation
ciblée -- calibration entière à refaire).
"""
import sys

import torch

sys.path.insert(0, ".")
from tests.test_awq_seuil_echantillons import (H, IK, _magnitudes_variees,
                                               _tiny_moe)

from acvram.engine.config import load_model_spec
from acvram.hardware.profiles import load_profile
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.calibrate import ActStats
from acvram.quant.convert import ConversionOptions, convert_checkpoint

E = 4   # nombre d'experts de _tiny_moe (test_awq_seuil_echantillons.py)


def _degenere(n):
    """Deux canaux extrêmes, les autres au plancher `clamp(min=1e-6)`
    (`calibrate.py:302`) -- le motif exact mesuré sur Nemotron (étendue
    1,7e6×), pas une simple magnitude uniforme."""
    m = torch.full((n,), 1e-6)
    m[0] = 50.0
    m[1] = 30.0
    return ActStats(m, m, 64)


def _stats_avec_n_malades(n_malades):
    """Les TROIS tenseurs (gate/up/down) de `n_malades` experts sur 4
    portent le motif dégénéré ; les autres experts un motif sain."""
    sain = {}
    for e in range(min(n_malades, E)):
        sain[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = _degenere(H)
        sain[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = _degenere(H)
        sain[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = _degenere(IK)
    for e in range(n_malades, E):
        sain[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = ActStats(
            _magnitudes_variees(H), None, 64)
        sain[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = ActStats(
            _magnitudes_variees(H), None, 64)
        sain[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = ActStats(
            _magnitudes_variees(IK), None, 64)
    return sain


def _convertir(tmp_path, n_malades, nom):
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    stats = _stats_avec_n_malades(n_malades)
    out = tmp_path / nom
    report = convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out)),
                                spec=spec, stats=stats)
    return out, report


def test_un_seul_expert_malade_est_replie_pas_refuse(tmp_path):
    """1/4 experts malades (3/17 tenseurs, 17,6 %, sous le plafond de 50 %) :
    la conversion ABOUTIT, les tenseurs fautifs sont repliés à l'identité,
    les 3 autres experts gardent leur AWQ. Un changement qui doit casser :
    retirer le repli (ne garder que le refus) fait lever une ValueError ici."""
    out, report = _convertir(tmp_path, n_malades=1, nom="un_malade")
    assert report.tenseurs_replies, "au moins un tenseur de l'expert 0 doit etre replie"
    assert all(".experts.0." in n for n in report.tenseurs_replies), (
        "seul l'expert 0 (le seul malade) doit etre replie")

    import json
    manifest = json.loads((out / "acvram_manifest.json").read_text())
    replies = manifest["tenseurs_replies_identite"]
    assert replies["nombre"] == len(report.tenseurs_replies)
    assert replies["part"] < 0.5
    for n in report.tenseurs_replies:
        entry = manifest["tensors"][n]
        assert 0.80 <= entry["ratio_norme"] <= 1.25, f"{n} replie : doit revenir dans les bornes"

    # temoin : l'expert sain voisin garde une vraie echelle AWQ (pas replie)
    from safetensors import safe_open
    for shard in sorted(out.glob("acvram-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            if "model.layers.0.mlp.experts.1.down_proj.weight.act_scale" in fh.keys():
                scale = fh.get_tensor("model.layers.0.mlp.experts.1.down_proj.weight.act_scale")
                assert not torch.allclose(scale, torch.ones_like(scale))


def test_majorite_de_tenseurs_malades_refuse_toujours(tmp_path, monkeypatch):
    """Au-dessus du plafond de 50 % du total de tenseurs vérifiés : plus
    une réparation ciblée, la conversion est REFUSÉE comme avant le
    repli. Le motif d'activation réel donne un taux de casse VARIABLE
    d'un tenseur à l'autre (dépend du poids source, pas seulement de la
    statistique) -- forcer `ratio_norme` directement pour une PROPORTION
    connue de tenseurs isole le plafond des 50 % de la stochasticité de
    la recherche AWQ elle-même (déjà couverte par le test ci-dessus et
    par `verdict-controle-parc-ratio-norme-17-09.md`)."""
    import itertools

    import acvram.quant.convert as conv_mod

    original = conv_mod.quantize_with_calibration
    compteur = itertools.count()

    def _force_mauvais(weight, fmt, stats=None, **kw):
        qt, scaler, metrics = original(weight, fmt, stats=stats, **kw)
        # 6 tenseurs sur les ~17 nvfp4 de ce jouet suffisent a depasser 50 % :
        # casser les 10 premiers appels avec AWQ reellement tente.
        if stats is not None and "ratio_norme" in metrics and next(compteur) < 10:
            metrics = {**metrics, "ratio_norme": 0.3}
        return qt, scaler, metrics

    monkeypatch.setattr(conv_mod, "quantize_with_calibration", _force_mauvais)

    import pytest
    with pytest.raises(ValueError, match="calibration est à refaire"):
        _convertir(tmp_path, n_malades=1, nom="tous_malades")
