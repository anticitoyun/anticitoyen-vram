"""g9m (01/10) : « hybride » = porte un état récurrent (`couches_recurrentes > 0`), jamais « a des layer_types ».

Avant le correctif, `est_hybride = bool(layer_types)` prenait gemma-3/4 (sliding/full, 0 récurrent) pour un hybride :
l'appariement du préfixe était plafonné à la plus grande frontière d'instantané photographiée — `_photographier` ne range
rien sans `gdn_states` — donc 0 jeton servi à une requête identique (gemma-4-31B, S1 du 01/10 : `cached_prompt_tokens` 0,
Devstral 19 %), préfill coupé à 256 et décodage sans lot spéculatif. Sur carte (verdict g9m 01/10) : 7 952/7 953 jetons
servis après correctif. Retenu par le chef comme définition (pas de variable de régime) : ces tests cassent si l'on remet
`bool(layer_types)`.
"""
import torch

import acvram.engine.runner as R
from acvram.engine.sampler import SamplingParams

TYPES_GEMMA = ["sliding_attention", "full_attention"] * 2        # typé, 0 récurrent
TYPES_HYBRIDE = ["linear_attention", "full_attention"] * 2       # Qwen3-Next : 2 récurrents


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


def test_predicat_hybride_est_l_etat_recurrent(converted):
    """Le prédicat seul : typé sans récurrence → dense ; récurrent → hybride ; sans layer_types → dense."""
    assert not _moteur(converted, []).est_hybride
    eng = _moteur(converted, TYPES_GEMMA)
    assert eng.spec.couches_recurrentes == 0 and not eng.est_hybride, "bool(layer_types) est revenu : gemma pris pour un hybride"
    eng = _moteur(converted, TYPES_HYBRIDE)
    assert eng.spec.couches_recurrentes == 2 and eng.est_hybride


def test_couches_typees_sans_recurrence_servent_le_cache_de_prefixe(converted):
    """Casse si l'on remet bool(layer_types) : le cache servi retombe à 0 (288 → 0, reproduit à sec le 01/10)."""
    temoin = _deux_fois(_moteur(converted, []))                                   # dense sans layer_types : référence
    assert temoin > 0
    servi = _deux_fois(_moteur(converted, TYPES_GEMMA))
    assert servi > 0, "0 jeton servi : le modèle typé est repris pour un hybride (frontière d'instantané absente)"
    assert servi == temoin, "un modèle typé sans état récurrent doit reprendre exactement comme un dense"


def test_un_vrai_hybride_ne_sert_pas_le_cache_par_le_prefixe_seul(converted):
    """Le correctif ne touche pas les hybrides : Qwen3-Next reste hybride (instantanés, pas d'appariement par blocs seuls)."""
    eng = _moteur(converted, TYPES_HYBRIDE)
    assert eng.est_hybride
    assert _deux_fois(eng) == 0
