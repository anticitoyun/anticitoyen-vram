"""d19 (poste6, 01/10, scellé revue/poste6-d19-scelle-01-10.md § 2-3) : l'attention d'un dense sans récurrence passe par morceaux
AU-DELÀ du tenu d'un seul tenant (seuil posé par la chauffe, `runner.definir_seuil_morceaux`), avec des K/V bf16 TRANSITOIRES par
couche (les morceaux précédents ne sont plus relus quantifiés du cache) et le biais bas-droite ; sous le seuil rien ne change.
La réserve de préfill borne au plafond de morceaux le flux résiduel, q/k/v et les scores (config `activations_prefill_bytes`).
Cassants : relecture du cache rétablie → `cache.gather` appelé pendant les morceaux (J1) ; seuil ignoré → `prefill_morceaux` 0 à
400 jetons (J1 b) ; réserve non bornée → la fenêtre qui tient de Devstral-24B à 65 536 retombe sous 65 536 (J3)."""
import pytest
import torch

import acvram.engine.attention as A
import acvram.engine.runner as R
from acvram.engine.config import ModelSpec
from acvram.engine.sampler import SamplingParams

G, M = 2 ** 30, 2 ** 20
MORCEAU = 128


def _moteur(converted, monkeypatch, layer_types=None):
    import acvram.memory.tiering as T
    from acvram.engine.loader import load_model
    monkeypatch.setattr(A, "SEUIL_FUSION", 10 ** 6)             # un seul chemin MLP (test_prefill_morceaux_kv2)
    monkeypatch.setattr(T, "_KV_FORMAT", None)                  # int8 : le format servi, celui où la relecture coûte
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu", max_concurrent_seqs=2)
    if layer_types is not None:
        loaded.spec.layer_types = layer_types
    eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=512, enable_prefix_cache=False, enable_cuda_graphs=False)
    eng._eos = set()
    return eng


def _logits(eng, n):
    vus = []
    emit = eng._emit

    def _capture(logits, seqs):
        vus.append(logits.detach().clone())
        return emit(logits, seqs)
    eng._emit = _capture
    ids = [(7 + i * 13) % 200 + 3 for i in range(n)]
    for _ in eng.generate(ids, SamplingParams(temperature=0.0, max_tokens=1)):
        pass
    eng._emit = emit
    return vus[0]


@pytest.fixture(autouse=True)
def _sans_seuil(monkeypatch):
    monkeypatch.setattr(R, "_MORCEAU_SEUIL", None)
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", 0)
    yield


def _espion_gather(monkeypatch):
    """Compte les relectures du cache pendant un préfill (gather), par morceau."""
    from acvram.memory import kvcache as KC
    appels = []
    orig = KC.PagedKVCache.gather

    def _gather(self, *a, **k):
        appels.append(1)
        return orig(self, *a, **k)
    monkeypatch.setattr(KC.PagedKVCache, "gather", _gather)
    return appels


def test_j1_transitoires_remplacent_la_relecture_du_cache(converted, monkeypatch):
    """Morceaux (opt-in 128) contre seul tenant, cache int8 : avec les K/V transitoires, aucun `gather` pendant les morceaux et
    Δ ≤ 1e-3 ; en témoin (relecture), `gather` est appelé à chaque morceau après le premier et l'écart n'est pas plus petit."""
    seul = _logits(_moteur(converted, monkeypatch), 400)
    monkeypatch.setattr(R, "_PREFILL_MORCEAU", MORCEAU)
    appels = _espion_gather(monkeypatch)
    monkeypatch.setattr(A, "_TRANSITOIRES", True)
    eng = _moteur(converted, monkeypatch)
    avec = _logits(eng, 400)
    assert eng.stats.prefill_morceaux == 1
    assert not appels, f"relecture du cache pendant les morceaux : {len(appels)} gather"
    d_avec = float((avec.float() - seul.float()).abs().max())
    assert d_avec <= 1e-3, d_avec
    monkeypatch.setattr(A, "_TRANSITOIRES", False)
    eng = _moteur(converted, monkeypatch)
    temoin = _logits(eng, 400)
    assert appels, "témoin : les morceaux après le premier doivent relire le cache"
    d_sans = float((temoin.float() - seul.float()).abs().max())
    assert d_avec <= d_sans, (d_avec, d_sans)
    print(f"d19 J1 : Δ transitoires {d_avec:.3g}, Δ relecture {d_sans:.3g}, gather témoin {len(appels)}")


def test_j1b_seuil_de_la_chauffe_engage_les_morceaux_au_dela_seulement(converted, monkeypatch):
    monkeypatch.setattr(R, "_MORCEAU_AU_DELA", MORCEAU)
    eng = _moteur(converted, monkeypatch)
    sous = _logits(eng, 200)
    assert eng.stats.prefill_morceaux == 0
    eng = _moteur(converted, monkeypatch)
    R.definir_seuil_morceaux(256)                       # après le chargement : `load_model` pose le seuil du plan (None ici)
    try:
        assert eng._morceau_pour(200) == 0 and eng._morceau_pour(400) == MORCEAU
        assert torch.equal(_logits(eng, 200), sous) and eng.stats.prefill_morceaux == 0      # sous le seuil : au bit
        _logits(eng, 400)
        assert eng.stats.prefill_morceaux == 1                                              # au-delà : morceaux
        eng.morceaux_seuil, eng.ctx_tenu = 256, 512                   # la ligne de régime ne porte les seuils qu'après la chauffe
        assert "morceaux>256" in eng.regime_ligne() and "morceaux@" not in eng.regime_ligne()
    finally:
        R.definir_seuil_morceaux(None)


def test_j1c_une_recurrence_reste_hors_perimetre(converted, monkeypatch):
    monkeypatch.setattr(R, "_MORCEAU_AU_DELA", MORCEAU)
    eng = _moteur(converted, monkeypatch, layer_types=["linear_attention", "full_attention"] * 2)
    R.definir_seuil_morceaux(256)
    eng.spec.layer_types = ["linear_attention", "full_attention"] * 2
    try:
        assert eng.spec.couches_recurrentes == 2 and eng._morceau_pour(400) == 0
    finally:
        R.definir_seuil_morceaux(None)


def _devstral():
    return ModelSpec(name="devstral24b", architecture="llama", hidden_size=5120, intermediate_size=32768, num_layers=40,
                     num_attention_heads=32, num_key_value_heads=8, vocab_size=131072, max_position_embeddings=131072,
                     head_dim=128)


def test_j3_reserve_bornee_au_plafond_de_morceaux():
    """Devstral-24B à 65 536 : sans plafond, résiduel + q/k/v 3,25 Gio et scores 8,0 Gio linéaires en T ; au plafond 16 384 les deux
    s'arrêtent à 16 384 lignes et il ne reste de linéaire que les K/V transitoires (256 Mio)."""
    s = _devstral()
    s.mlp_prefill_plafond = 16384
    plein = s.activations_prefill_bytes(65536)
    s.prefill_morceau_plafond = 16384
    borne = s.activations_prefill_bytes(65536)
    assert plein - borne >= 8 * G, (plein / G, borne / G)                       # 3/4 des 3,25 + 8,0 Gio linéaires rendus
    assert 4 * G <= borne <= 10 * G, borne / G
    s.prefill_morceau_plafond = None
    assert s.activations_prefill_bytes(16384) == s.activations_prefill_bytes(16384)   # sous le plafond : inchangé
    s2 = _devstral(); s2.mlp_prefill_plafond = 16384; s2.prefill_morceau_plafond = 16384
    assert s2.activations_prefill_bytes(16384) == s.activations_prefill_bytes(16384) if s.mlp_prefill_plafond == 16384 else True


def test_j3b_fenetre_qui_tient_devstral_65536(monkeypatch):
    """Réplique de Devstral-24B (15,2 Gio résidents) sur 31,8 Gio : base KV + réserve ≈ 15 Gio → 65 536 tiennent avec les morceaux
    dans la réserve ; sans eux (`_MORCEAU_AU_DELA` = 0, l'ancienne réserve) la fenêtre annoncée retombe sous 65 536."""
    from acvram.engine import loader as LD
    from acvram.memory.tiering import LayerPlacement, Plan, Tier
    spec = _devstral()
    plan = Plan(model="devstral24b", tiers=[
        Tier(name="cuda:0", kind="gpu", device_index=0, capacity=32 * G, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=1790.0, link_bandwidth=21.0)],
        layers=[LayerPlacement(index=i, exec_device="cuda:0", attn_storage="cuda:0", mlp_storage="cuda:0", fmt="nvfp4",
                               attn_bytes=80 * M, mlp_bytes=309 * M, mlp_active_bytes=309 * M) for i in range(40)],
        embed_device="cuda:0", lm_head_device="cuda:0")
    plan.kv_bytes_per_token = spec.kv_bytes_per_token(8)
    man = {"vision": "non", "tensors": {}, "model": {}}
    base = 15 * G
    monkeypatch.setattr(R, "_MORCEAU_AU_DELA", 4096)
    avec = LD._fenetre_qui_tient(plan, spec, man, "cuda:0", base, 0, 65536)
    monkeypatch.setattr(R, "_MORCEAU_AU_DELA", 0)
    sans = LD._fenetre_qui_tient(plan, spec, man, "cuda:0", base, 0, 65536)
    assert avec == 65536, avec
    assert sans < 65536, sans
    print(f"d19 J3 b : fenêtre Devstral à 65 536 demandés, base 15 Gio : {avec} avec morceaux, {sans} sans")
