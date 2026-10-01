"""Levier 2 (poste6, 01/10, scellé revue/poste6-gemma-anneau-scelle-01-10.md § 3, A1-A5) : cache KV en ANNEAU de R blocs par
séquence pour les couches à fenêtre glissante. Jouet de la CI : 4 couches, les couches 0 et 2 à fenêtre 64 (R = 7), bloc 16, cache
int8 ; invite de 400 jetons (> 5 fenêtres) puis 64 pas de décodage glouton.
A1 : anneau contre plein AU BIT (ids et logits) — mêmes noyaux, mêmes positions lues, mêmes K/V. A2 (cassant) : un anneau trop petit
(R = 4) diverge. A3 : octets d'une séquence et fenêtre qui tient d'une réplique de gemma-4-31B (30,2 → 5,45 Gio ; 65 536 sous l'anneau).
A5 : sous l'anneau le cache de préfixe est coupé (0 jeton servi, dit)."""
import contextlib
import io

import pytest
import torch

import acvram.engine.runner as R
from acvram.engine.config import ModelSpec
from acvram.engine.sampler import SamplingParams
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

G, M = 2 ** 30, 2 ** 20
FENETRE = 64
COUCHES_FENETRE = (0, 2)


def _moteur(converted, anneau: int, prefixe: bool = False):
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cpu", max_concurrent_seqs=2)
    spec = loaded.spec
    spec.layer_types = ["sliding_attention" if i in COUCHES_FENETRE else "full_attention" for i in range(len(loaded.model.layers))]
    spec.sliding_window = FENETRE
    for i, layer in enumerate(loaded.model.layers):
        layer.self_attn.window = FENETRE if i in COUCHES_FENETRE else 0
    spec.kv_anneau = anneau
    if anneau:
        for i in COUCHES_FENETRE:
            c = loaded.model.caches[i]
            loaded.model.caches[i] = PagedKVCache(KVCacheConfig(
                num_layers=1, num_kv_heads=c.cfg.num_kv_heads, head_dim=c.cfg.head_dim, num_blocks=2 * anneau,
                dtype=c.cfg.dtype, device="cpu", anneau=anneau))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        eng = R.Engine(loaded, None, max_batch_size=2, max_model_len=512, enable_prefix_cache=prefixe, enable_cuda_graphs=False)
    eng._eos = set()
    eng._journal_init = out.getvalue()
    return eng


def _generer(eng, n=400, pas=64):
    vus = []
    emit = eng._emit

    def _capture(logits, seqs):
        vus.append(logits.detach().clone().float())
        return emit(logits, seqs)
    eng._emit = _capture
    ids = [(7 + i * 13) % 200 + 3 for i in range(n)]
    jetons = []
    for o in eng.generate(ids, SamplingParams(temperature=0.0, max_tokens=pas)):
        jetons += list(o.token_ids)
    eng._emit = emit
    return jetons, vus


def test_a1_anneau_au_bit_du_plein(converted):
    R_ = _moteur(converted, 0).spec.anneau_R()
    assert R_ == 7, R_
    ids_p, lp = _generer(_moteur(converted, 0))
    eng = _moteur(converted, R_)
    assert "(anneau R=7, préfixe off)" in eng.regime_ligne()
    ids_a, la = _generer(eng)
    assert len(ids_p) == 64 and ids_a == ids_p, (ids_p[:8], ids_a[:8])
    assert len(lp) == len(la) and all(torch.equal(a, b) for a, b in zip(lp, la)), \
        max(float((a - b).abs().max()) for a, b in zip(lp, la))
    assert eng._anneau_libres and all(s.anneau_slot == -1 for s in eng.running + eng.waiting)   # créneau rendu


def test_a2_anneau_trop_petit_diverge(converted):
    """Cassant : R = 4 < fenêtre/16 + 2 — le décodage relit des blocs recyclés ; si ce test passait, A1 ne contrôlerait rien."""
    ids_p, _ = _generer(_moteur(converted, 0))
    ids_a, _ = _generer(_moteur(converted, 4))
    assert ids_a != ids_p


def test_a5_cache_de_prefixe_coupe_sous_l_anneau(converted):
    eng = _moteur(converted, 7, prefixe=True)
    assert "cache de préfixe coupé" in eng._journal_init and not eng.allocator.enable_prefix_cache
    ids = [(7 + i * 13) % 200 + 3 for i in range(300)]
    for _ in range(2):
        for _ in eng.generate(ids, SamplingParams(temperature=0.0, max_tokens=1)):
            pass
    assert eng.stats.cached_prompt_tokens == 0
    temoin = _moteur(converted, 0, prefixe=True)
    for _ in range(2):
        for _ in temoin.generate(ids, SamplingParams(temperature=0.0, max_tokens=1)):
            pass
    assert temoin.stats.cached_prompt_tokens > 0                 # sans anneau (≤ tenu plein) : inchangé


def _gemma():
    return ModelSpec(name="gemma31b", architecture="llama", hidden_size=5376, intermediate_size=21504, num_layers=60,
                     num_attention_heads=32, num_key_value_heads=16, vocab_size=262144, max_position_embeddings=262144,
                     head_dim=256, sliding_window=1024, layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10)


def test_a3_octets_et_fenetre_qui_tient_gemma():
    from acvram.engine import loader as LD
    from acvram.memory.tiering import LayerPlacement, Plan, Tier
    s = _gemma()
    assert len(s.couches_fenetre) == 50 and s.anneau_R() == 67
    plein = s.kv_bytes_pour_sequence(65536, anneau=0)
    anneau = s.kv_bytes_pour_sequence(65536, anneau=67)
    assert 30.0 * G <= plein <= 30.5 * G and 5.3 * G <= anneau <= 5.6 * G, (plein / G, anneau / G)
    plan = Plan(model="gemma31b", tiers=[
        Tier(name="cuda:0", kind="gpu", device_index=0, capacity=32 * G, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=1790.0, link_bandwidth=21.0)],
        layers=[LayerPlacement(index=i, exec_device="cuda:0", attn_storage="cuda:0", mlp_storage="cuda:0", fmt="nvfp4",
                               attn_bytes=74 * M, mlp_bytes=195 * M, mlp_active_bytes=195 * M) for i in range(60)],
        embed_device="cuda:0", lm_head_device="cuda:0")
    plan.kv_bytes_per_token = s.kv_bytes_per_token(8)
    man = {"vision": "non", "tensors": {}, "model": {}}
    base = int(10.4 * G)                                             # 29,5 libres − 19,1 de poids
    s.kv_anneau = 0
    sans = LD._fenetre_qui_tient(plan, s, man, "cuda:0", base, 0, 65536)
    s.kv_anneau = 67
    assert LD._kv_plancher(plan, s, 65536, "cuda:0") == anneau
    avec = LD._fenetre_qui_tient(plan, s, man, "cuda:0", base, 0, 65536)
    # la formule garde le terme de scores d'un seul tenant à fenêtre en T (mesure du 01/10 18 h : 3,91 Gio à 16 384) : elle annonce
    # 34 816 sous l'anneau à 10,4 Gio de base ; sur carte la chauffe a tenu 65 536 (morceaux au-delà du tenu, pic 1,93 Gio) — c'est elle
    # qui prouve, la formule reste prudente ; sans anneau : 15 360
    assert sans < avec and avec >= 32768, (sans, avec)
    print(f"levier 2 A3 : gemma 65 536 — plein {plein / G:.2f} Gio, anneau {anneau / G:.2f} ; fenêtre qui tient sans {sans}, avec {avec}")
