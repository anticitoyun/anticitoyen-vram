"""22/09 (verdict-nsys-familles) : la ligne de régime disait `chemin_moe=mma-a4`
(l env) alors que le pas servi était `gemv_marlin` — deux modules conçus
pour un chemin mort. La ligne porte désormais le chemin ATTEINT, compté par
`MoEBlock._chemin` (REGLES § 7) ; ce test casse si l env et le chemin
divergent sans que la ligne le dise."""
from types import SimpleNamespace

from acvram.engine.model import MoEBlock
from acvram.engine.moe import inertes_disposition_unique
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


def test_inertes_disposition_unique(monkeypatch):
    """Exerce le VRAI code de moe.py (pas une valeur injectée) : casse si l'ajout est retiré."""
    monkeypatch.delenv("ACVRAM_MOE_MMA", raising=False)
    monkeypatch.delenv("ACVRAM_PREFILL_DEQUANT", raising=False)
    assert inertes_disposition_unique(True) == set()
    assert inertes_disposition_unique(False) == set()          # non unique : jamais inerte, même posée
    monkeypatch.setenv("ACVRAM_MOE_MMA", "0")
    assert inertes_disposition_unique(True) == {"moe_mma=0(inerte:disposition unique)"}
    assert inertes_disposition_unique(False) == set()
    monkeypatch.delenv("ACVRAM_MOE_MMA")
    monkeypatch.setenv("ACVRAM_PREFILL_DEQUANT", "1")
    assert inertes_disposition_unique(True) == {"prefill_dequant=1(inerte:disposition unique)"}
    monkeypatch.setenv("ACVRAM_MOE_MMA", "0")
    assert inertes_disposition_unique(True) == {"moe_mma=0(inerte:disposition unique)",
                                                  "prefill_dequant=1(inerte:disposition unique)"}


def test_la_ligne_dit_l_inertie(converted, monkeypatch):
    """Pièce 127 (poste6, verdict-piece125-24-09) : ACVRAM_MOE_MMA/ACVRAM_PREFILL_DEQUANT posées mais
    sans effet sur la disposition unique (moe.py:791-798) — la ligne le dit, au lieu de rester muette
    comme si le réglage avait agi (faute « flag no-op »). Casse si l'inertie redevient muette."""
    from test_engine import _engine_cpu
    monkeypatch.setenv("ACVRAM_MOE_MMA", "0")
    eng = _engine_cpu(converted)
    faux = MoEBlock.__new__(MoEBlock)
    faux.__dict__.update({"chemins": {"marlin": 1}, "_stack_state": "oui", "experts": [], "_raison_repli": "",
                          "_inertes": {"moe_mma=0(inerte:disposition unique)"}})
    modules = list(eng.model.modules()) + [faux]
    monkeypatch.setattr(eng.model, "modules", lambda: modules)
    ligne = eng.regime_ligne()
    assert "moe_mma=0(inerte:disposition unique)" in ligne, ligne
    faux.__dict__["_inertes"] = set()
    assert "inerte" not in eng.regime_ligne()
