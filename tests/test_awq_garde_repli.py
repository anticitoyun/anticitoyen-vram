"""poste7-awq-relu2-garde-repli-17-09 : geste (b). La garde de norme/étendue
(commit 2a68aa6, corrigée par poste7-awq-plancher-median-faute-18-09)
refusait TOUTE la conversion dès qu'un seul tenseur sortait de
[0,80;1,25] (ratio de norme) ou 4096 (étendue des échelles) -- exercé en
vrai sur Nemotron calibA le 17/09 : 145 tenseurs `mlp.experts.*.down_proj`
hors bornes malgré le seuil de 8 échantillons, cause réelle identifiée
ailleurs (ReLU² creux, pas le nombre d'échantillons). Le refus total
laissait aucune sortie utilisable pour un défaut LOCALISÉ. Ce fichier
teste le repli PAR TENSEUR : identité pour le(s) tenseur(s) fautif(s),
conversion qui aboutit, sous un plafond de 50 % (au-delà, plus une
réparation ciblée -- calibration entière à refaire).

Le plancher corrigé (`_magnitude_avec_plancher_relatif`, borné sur le
MAXIMUM du tenseur) rend l'étendue ≤ 4096 PAR CONSTRUCTION pour toute
statistique réelle passée par la recherche AWQ (vérifié dans
`test_awq_plancher_relatif.py`) -- un motif organique (un canal extrême,
les autres écrasés) ne suffit donc plus à faire sortir le ratio ou
l'étendue des bornes. Forcer directement les métriques via un monkeypatch
isole le test du MÉCANISME de repli/plafond de la recherche AWQ
elle-même, déjà couverte séparément.
"""
import itertools
import sys

sys.path.insert(0, ".")
from tests.test_awq_seuil_echantillons import E, H, IK, _magnitudes_variees, _tiny_moe

import acvram.quant.convert as conv_mod
from acvram.engine.config import load_model_spec
from acvram.hardware.profiles import load_profile
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.calibrate import ActStats
from acvram.quant.convert import ConversionOptions, convert_checkpoint


def _stats_toutes_saines():
    """Une statistique réelle (donc `st is not None`) pour chaque tenseur
    d'expert -- sans quoi AUCUN tenseur ne passe par la recherche AWQ que
    `_force_n_mauvais` cible, et le monkeypatch ne corrompt jamais rien."""
    sain = {}
    for e in range(E):
        sain[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = ActStats(
            _magnitudes_variees(H), None, 64)
        sain[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = ActStats(
            _magnitudes_variees(H), None, 64)
        sain[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = ActStats(
            _magnitudes_variees(IK), None, 64)
    return sain


def _force_n_mauvais(monkeypatch, n):
    """Force les `n` PREMIERS tenseurs quantifiés avec de vraies
    statistiques (AWQ réellement tenté) à porter un ratio de norme hors
    bornes -- déterministe, indépendant du poids source ou de la
    statistique fournie."""
    original = conv_mod.quantize_with_calibration
    compteur = itertools.count()

    def _wrapper(weight, fmt, stats=None, **kw):
        qt, scaler, metrics = original(weight, fmt, stats=stats, **kw)
        if stats is not None and "ratio_norme" in metrics and next(compteur) < n:
            metrics = {**metrics, "ratio_norme": 0.3}
        return qt, scaler, metrics

    monkeypatch.setattr(conv_mod, "quantize_with_calibration", _wrapper)


def _convertir(tmp_path, nom):
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    out = tmp_path / nom
    report = convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out)),
                                spec=spec, stats=_stats_toutes_saines())
    return out, report


def test_une_minorite_de_tenseurs_malades_est_repliee_pas_refusee(tmp_path, monkeypatch):
    """2 tenseurs forcés mauvais sur ~17-22 au total (largement sous le
    plafond de 50 %) : la conversion ABOUTIT, ces deux tenseurs sont
    repliés à l'identité (ratio de norme revenu dans les bornes), le
    reste garde son AWQ. Un changement qui doit casser : retirer le repli
    (ne garder que le refus) fait lever une ValueError ici."""
    _force_n_mauvais(monkeypatch, n=2)
    out, report = _convertir(tmp_path, "minorite")
    assert len(report.tenseurs_replies) == 2

    import json
    manifest = json.loads((out / "acvram_manifest.json").read_text())
    replies = manifest["tenseurs_replies_identite"]
    assert replies["nombre"] == 2
    assert replies["part"] < 0.5
    for n in report.tenseurs_replies:
        entry = manifest["tensors"][n]
        assert 0.80 <= entry["ratio_norme"] <= 1.25, f"{n} replie : doit revenir dans les bornes"


def test_majorite_de_tenseurs_malades_refuse_toujours(tmp_path, monkeypatch):
    """10 tenseurs forcés mauvais (au-dessus du plafond de 50 % du total
    de ce jouet, ~17-22 tenseurs nvfp4) : plus une réparation ciblée, la
    conversion est REFUSÉE comme avant le repli."""
    _force_n_mauvais(monkeypatch, n=10)
    import pytest
    with pytest.raises(ValueError, match="calibration est à refaire"):
        _convertir(tmp_path, "majorite")
