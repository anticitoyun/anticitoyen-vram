"""P3, veille TRT-LLM/FlashInfer (Sage, 14/09) : taux de réutilisation du
préfixe (`hit_rate`, `cached_prompt_tokens`, jetons remontés de l'hôte)
publiés dans `/metrics` (EngineStats.to_dict) et `regime_ligne`. Pur CPU :
EngineStats est un dataclass simple, pas besoin de carte.
"""
from acvram.engine.runner import BLOCK_SIZE, EngineStats


def test_hit_rate_zero_sans_trafic():
    s = EngineStats()
    assert s.hit_rate == 0.0


def test_hit_rate_calcule_correctement():
    s = EngineStats(cached_prompt_tokens=300, prefill_tokens=100)
    assert s.hit_rate == 0.75


def test_host_kv_tokens_depuis_kv_refills():
    s = EngineStats(kv_refills=5)
    assert s.host_kv_tokens == 5 * BLOCK_SIZE


def test_to_dict_publie_les_trois_champs():
    s = EngineStats(cached_prompt_tokens=300, prefill_tokens=100, kv_refills=2)
    d = s.to_dict()
    assert d["hit_rate"] == 0.75
    assert d["cached_prompt_tokens"] == 300
    assert d["host_kv_tokens"] == 2 * BLOCK_SIZE
    assert d["kv_refills"] == 2
