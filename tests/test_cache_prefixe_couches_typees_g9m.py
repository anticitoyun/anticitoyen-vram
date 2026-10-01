"""g9m (01/10) : un modèle à couches typées SANS état récurrent (gemma-3/4 : sliding/full) doit servir son cache de préfixe.

Avant le correctif, `est_hybride = bool(layer_types)` le prenait pour un hybride : l'appariement du préfixe était plafonné à la
plus grande frontière d'instantané photographiée — `_photographier` ne range rien sans `gdn_states` — donc 0 jeton servi à une
requête identique (gemma-4-31B, S1 du 01/10 : `cached_prompt_tokens` 0, Devstral 19 %). Reproduit à sec : 288 → 0.
"""
import pytest
import torch

import acvram.engine.runner as R
from acvram.engine.sampler import SamplingParams


def _moteur(converted, layer_types):
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu", max_concurrent_seqs=2)
    loaded.spec.layer_types = layer_types
    eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=512, enable_prefix_cache=True, enable_cuda_graphs=False)
    eng._eos = set()
    return eng


def _deux_fois(eng, n=300):
    ids = [(7 + i * 13) % 200 + 3 for i in range(n)]
    for _ in range(2):
        for _ in eng.generate(ids, SamplingParams(temperature=0.0, max_tokens=1)):
            pass
    return eng.stats.cached_prompt_tokens


def test_couches_typees_sans_recurrence_servent_le_cache_de_prefixe(converted, monkeypatch):
    """Les DEUX chemins (chef 01/10 : variable de régime, défaut = ancien comportement) : à 0, gemma-4 est hybride et ne sert
    rien ; à 1, il reprend comme un dense. Le défaut basculera sur la garde qualité de poste2."""
    temoin = _deux_fois(_moteur(converted, []))                                   # dense sans layer_types : référence
    assert temoin > 0
    types = ["sliding_attention", "full_attention"] * 2                            # gemma-4 : typé, 0 récurrent
    monkeypatch.setattr(R, "_HYBRIDE_PAR_RECURRENCE", False)                      # défaut : ancien comportement
    eng = _moteur(converted, types)
    assert eng.spec.couches_recurrentes == 0 and eng.est_hybride
    assert _deux_fois(eng) == 0, "défaut : un modèle typé est encore pris pour un hybride, 0 jeton servi (dit, pas un bogue du test)"
    monkeypatch.setattr(R, "_HYBRIDE_PAR_RECURRENCE", True)                       # ACVRAM_HYBRIDE_PAR_RECURRENCE=1
    eng = _moteur(converted, types)
    assert not eng.est_hybride
    assert _deux_fois(eng) == temoin, "un modèle typé sans état récurrent doit reprendre comme un dense"


@pytest.mark.parametrize("par_recurrence", [False, True])
def test_un_vrai_hybride_reste_hybride(converted, monkeypatch, par_recurrence):
    from acvram.engine.loader import load_model
    monkeypatch.setattr(R, "_HYBRIDE_PAR_RECURRENCE", par_recurrence)
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu", max_concurrent_seqs=2)
    loaded.spec.layer_types = ["linear_attention", "full_attention"] * 2
    assert loaded.spec.couches_recurrentes == 2
    eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=512, enable_prefix_cache=True, enable_cuda_graphs=False)
    assert eng.est_hybride
