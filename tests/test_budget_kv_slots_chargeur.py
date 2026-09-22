"""Budget KV dimensionné pour le VRAI nombre de séquences, pas pour un
défaut caché (Océane, à sec, `sage-poste-d-verdict-17-09.md` § 2).

Laure a trouvé qu'`Engine._replanifier` (`loader.py`) ne passait que
`max_model_len` au planificateur — jamais `max_concurrent_seqs` — donc le
budget KV tournait TOUJOURS pour 8 séquences (`PlannerOptions` par défaut),
quel que soit le `b` réellement demandé. À b=12, 4 séquences sur 12
tronquaient silencieusement avant `max_tokens` (`Engine._finish_budget_epuise`,
budget épuisé) : le débit certifié mesurait un lot plus petit que celui
annoncé, sans qu'aucun compteur ne le signale.

`detect_rig` est monkeypatché sur le profil `target_rig` (comme la fixture
`converted` de conftest.py) : on veut un rig déterministe, pas la machine
réelle qui exécute la suite.
"""
from unittest.mock import patch

import pytest
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
from acvram.memory.tiering import PlannerOptions


@pytest.fixture
def rig_fige(target_rig):
    with patch("acvram.hardware.detect.detect_rig", return_value=target_rig):
        yield target_rig


def _charger(converted, rig_fige, slots):
    kwargs = {} if slots is None else {"max_concurrent_seqs": slots}
    return load_model(converted, dtype=torch.bfloat16, max_model_len=512,
                      device_override="cpu", **kwargs)


def test_budget_kv_suit_les_slots_demandes(converted, rig_fige):
    """Le budget grandit avec le nombre de séquences demandé — s'il ne
    bougeait pas entre 1, 8 et 12, ce serait le symptôme exact du bogue :
    un défaut fixe qui ignore l'appelant."""
    budgets = {s: _charger(converted, rig_fige, s).plan.kv_max_tokens
              for s in (1, 8, 12)}
    assert budgets[1] < budgets[8] < budgets[12], budgets


def test_kv_planned_seqs_porte_le_slot_demande(converted, rig_fige):
    for slots in (1, 8, 12):
        loaded = _charger(converted, rig_fige, slots)
        assert loaded.plan.kv_planned_seqs == slots


def test_sans_max_concurrent_seqs_retombe_sur_le_manifeste_pas_sur_8(converted, rig_fige):
    """`converted` (conftest.py) a été planifié avec max_concurrent_seqs=2 :
    un appelant qui ne dit rien doit hériter cette intention, pas retomber
    sur le défaut 8 de `PlannerOptions` — même logique que `max_model_len`
    (docstring de `_replanifier`)."""
    loaded = _charger(converted, rig_fige, None)
    assert loaded.plan.kv_planned_seqs == 2
    assert loaded.plan.kv_planned_seqs != PlannerOptions().max_concurrent_seqs


def test_budget_pour_12_depasse_celui_du_manifeste_a_2(converted, rig_fige):
    """Le cas réel du bogue : demander explicitement plus de séquences que
    le manifeste n'en prévoyait doit AUGMENTER le budget, pas le laisser
    identique à celui planifié pour 2."""
    manifeste = _charger(converted, rig_fige, None)
    a_12 = _charger(converted, rig_fige, 12)
    assert a_12.plan.kv_max_tokens > manifeste.plan.kv_max_tokens


# ---------------------------------------------------------------------------
# regime_ligne() et le compteur de troncature
# ---------------------------------------------------------------------------

def test_regime_ligne_porte_kv_budget(converted, rig_fige):
    loaded = _charger(converted, rig_fige, 8)
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=512,
                    enable_cuda_graphs=False)
    ligne = engine.regime_ligne()
    assert f"kv_budget={engine.allocator.num_blocks * BLOCK_SIZE}/8" in ligne


def test_max_batch_size_au_dela_du_plan_leve_sage_reprise_18_09(converted, rig_fige):
    """La faille fermée le 18/09 (sage-reprise-ordre-18-09 §Suite,
    verdict-kv-budget-8-a-sec-18-09) : `load_model(max_model_len=…)` SEUL —
    sans `max_concurrent_seqs` — retombe sur le défaut hérité du manifeste
    (`converted` : 2, cf. `test_sans_max_concurrent_seqs_retombe_sur_le_manifeste_pas_sur_8`
    ci-dessus). Construire l'`Engine` à `max_batch_size=12` par-dessus ce
    plan est exactement le geste qui tronquait en silence — il doit
    maintenant lever, pas se lancer."""
    loaded = _charger(converted, rig_fige, None)
    assert loaded.plan.kv_planned_seqs == 2
    with pytest.raises(ValueError, match="max_batch_size=12"):
        Engine(loaded, None, max_batch_size=12, max_model_len=512,
              enable_cuda_graphs=False)


def test_max_batch_size_au_dela_du_plan_override_force_le_lancement(converted, rig_fige, monkeypatch):
    monkeypatch.setenv("ACVRAM_KV_PLAN_OVERRIDE", "1")
    loaded = _charger(converted, rig_fige, None)
    engine = Engine(loaded, None, max_batch_size=12, max_model_len=512,
                    enable_cuda_graphs=False)
    assert "kv_plan_override=1" in engine.regime_ligne()


def test_sequences_tronquees_budget_compte_et_apparait_dans_regime(converted, rig_fige, capsys):
    loaded = _charger(converted, rig_fige, 8)
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=512,
                    enable_cuda_graphs=False)
    engine.allocator = BlockAllocator(3, True)   # budget minuscule, force l'épuisement
    produced = list(engine.generate([5, 42, 7],
                                    SamplingParams(temperature=0.0, max_tokens=64)))
    assert produced, "aucune sortie produite"
    assert engine.stats.sequences_tronquees_budget >= 1
    assert engine.stats.to_dict()["sequences_tronquees_budget"] == engine.stats.sequences_tronquees_budget
    assert produced[-1].finished and produced[-1].finish_reason == "length"
    assert "budget KV épuisé" in capsys.readouterr().out
