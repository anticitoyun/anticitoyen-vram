"""22/09 (verdict-nsys-familles) : la ligne de régime disait `chemin_moe=mma-a4`
(l env) alors que le pas servi était `gemv_marlin` — deux modules conçus
pour un chemin mort. La ligne porte désormais le chemin ATTEINT, compté par
`MoEBlock._chemin` (REGLES § 7) ; ce test casse si l env et le chemin
divergent sans que la ligne le dise."""
from types import SimpleNamespace

from acvram.engine.model import MoEBlock
from acvram.engine.runner import chemin_moe_atteint


def test_chemin_atteint_depuis_les_compteurs():
    assert chemin_moe_atteint([]) == "atteint=non-atteint"
    assert chemin_moe_atteint([{"mma": 3, "marlin": 2}]) == "atteint=non-atteint"        # préfill seul
    assert chemin_moe_atteint([{"gemv_marlin": 48}, {"gemv_marlin": 48, "mma": 1}]) == "atteint=gemv_marlin"
    assert chemin_moe_atteint([{"gemv_marlin": 10, "decode_mma": 40}]) == "atteint=decode_mma+gemv_marlin"
    assert "decode_mma" in MoEBlock.CHEMINS_DECODAGE and "gemv_marlin" in MoEBlock.CHEMINS_DECODAGE


def test_la_ligne_dit_la_divergence(converted, monkeypatch):
    """env mma-a4 mais compteurs gemv_marlin → la ligne porte les deux, en clair."""
    from test_engine import _engine_cpu
    monkeypatch.setenv("ACVRAM_MOE_MMA", "1")
    eng = _engine_cpu(converted)
    faux = MoEBlock.__new__(MoEBlock)
    faux.__dict__.update({"chemins": {"gemv_marlin": 48}, "_stack_state": "oui", "experts": [], "_raison_repli": ""})
    modules = list(eng.model.modules()) + [faux]
    monkeypatch.setattr(eng.model, "modules", lambda: modules)
    ligne = eng.regime_ligne()
    assert "chemin_moe=mma-a4(atteint=gemv_marlin)" in ligne, ligne
    faux.__dict__["chemins"] = {}
    assert "chemin_moe=mma-a4(atteint=non-atteint)" in eng.regime_ligne()
