"""Pièce 104 — cache KV k8v4 (K int8 par jeton, V int4 par groupe de 32), à sec.

Chaque test a été cassé une fois en réintroduisant la faute qu'il garde
(scratchpad/poste5-p104-23-09/casser.py ; carnet poste5.md 23/09) : ordre des
quartets, échelle /15, clamp à −8, sentinelle, plancher, budget d'octets.
Les tests sur carte (noyau = jumeau au bit, lecteur = référence ± 2⁻⁷) sont
dans test_kv_k8v4_carte.py.
"""
import re

import pytest
import torch

from acvram.memory import kv_k8v4
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

HKV, D = 4, 128


def _cfg(num_blocks=8, **kw):
    return KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=num_blocks,
                         dtype="k8v4", device="cpu", **kw)


def test_quartets_ordre_pair_bas():
    codes = torch.tensor([[1, -2, 7, -7, 0, 3, -1, 5]], dtype=torch.int8)
    e = kv_k8v4.emballer(codes)
    assert e.dtype == torch.uint8 and e.shape == (1, 4)
    assert e[0, 0].item() == (1 & 0xF) | ((-2 & 0xF) << 4)      # pair en bas, impair en haut
    assert torch.equal(kv_k8v4.deballer(e), codes)


def test_quantifier_forme_et_echelles():
    v = torch.randn(5, HKV, D, dtype=torch.bfloat16)
    codes, sv = kv_k8v4.quantifier_v(v)
    assert codes.shape == (5, HKV, D // 2) and codes.dtype == torch.uint8
    assert sv.shape == (5, HKV, D // 32) and sv.dtype == torch.float16
    amax = v.float().reshape(5, HKV, 4, 32).abs().amax(-1)
    assert torch.equal(sv, (amax / 7.0).to(torch.float16).float().clamp(min=2 ** -24).to(torch.float16))


def test_erreur_bornee_demi_echelle():
    # l'échelle stockée est celle qui a quantifié : |v − v̂| ≤ sv/2 exactement
    v = torch.randn(64, HKV, D, dtype=torch.bfloat16) * 3
    codes, sv = kv_k8v4.quantifier_v(v)
    d = kv_k8v4.dequantifier_v(codes, sv, torch.float32)
    borne = sv.float().repeat_interleave(32, dim=-1) * (0.5 + 2 ** -20)
    assert ((v.float() - d).abs() <= borne).all()


def test_saturation_jamais_moins_huit():
    v = torch.zeros(1, 1, D, dtype=torch.bfloat16)
    v[0, 0, :32] = -4.0                        # amax = 4 → sv = 4/7, v/sv = −7 exactement
    v[0, 0, 3] = 4.0
    codes = kv_k8v4.deballer(kv_k8v4.quantifier_v(v)[0])
    assert codes.min().item() == -7 and codes.max().item() == 7
    assert (codes[0, 0, :32] != -8).all()


def test_echelle_groupe_zero_sans_nan():
    v = torch.zeros(3, HKV, D, dtype=torch.bfloat16)
    codes, sv = kv_k8v4.quantifier_v(v)
    assert (kv_k8v4.deballer(codes) == 0).all()
    assert (sv.float() == 2 ** -24).all()      # plancher : jamais 0 en half
    d = kv_k8v4.dequantifier_v(codes, sv, torch.float32)
    assert torch.isfinite(d).all() and (d == 0).all()


def test_cache_ecrit_relit_gather_egal_gather_fixed():
    cfg = _cfg()
    cache = PagedKVCache(cfg)
    assert cache.v.shape == (8, 16, HKV, D // 2) and cache.v.dtype == torch.uint8
    assert cache.v_scale.shape == (8, 16, HKV, 4)
    torch.manual_seed(0)
    k = torch.randn(20, HKV, D, dtype=torch.bfloat16)
    v = torch.randn(20, HKV, D, dtype=torch.bfloat16)
    slots = torch.arange(20)                   # blocs 0 et 1 (16 + 4 jetons)
    cache.write(slots, k, v)
    kk, vv = cache.gather(torch.tensor([0, 1]), 20, torch.float32)
    codes, sv = kv_k8v4.quantifier_v(v)
    assert torch.equal(vv, kv_k8v4.dequantifier_v(codes, sv, torch.float32))
    kf, vf = cache.gather_fixed(torch.tensor([[0, 1]]), torch.float32)
    assert torch.equal(vf[0, :20], vv) and torch.equal(kf[0, :20], kk)
    # K : chemin int8 par jeton inchangé
    assert cache.k.dtype == torch.int8 and cache.k_scale.shape == (8, 16, HKV)


def test_sentinelle_slot_negatif_n_ecrit_rien():
    cache = PagedKVCache(_cfg())
    avant = (cache.v.clone(), cache.v_scale.clone(), cache.k.clone())
    k = torch.randn(3, HKV, D, dtype=torch.bfloat16)
    v = torch.randn(3, HKV, D, dtype=torch.bfloat16)
    cache.write(torch.tensor([-1, -1, -1]), k, v)
    assert torch.equal(cache.v, avant[0]) and torch.equal(cache.v_scale, avant[1])
    assert torch.equal(cache.k, avant[2])


def test_lot_mele_delta_zero():
    """La même séquence, seule (b=1) ou dans un lot de 3 : sortie identique au bit."""
    from acvram.engine.layers import decode_attention_fixed
    torch.manual_seed(1)
    cache = PagedKVCache(_cfg(num_blocks=12))
    lens = [37, 5, 21]
    tables, slot = [], 0
    for i, n in enumerate(lens):
        blocs = list(range(i * 4, i * 4 + 4))
        tables.append(blocs)
        s = torch.tensor([blocs[j // 16] * 16 + j % 16 for j in range(n)])
        cache.write(s, torch.randn(n, HKV, D, dtype=torch.bfloat16),
                    torch.randn(n, HKV, D, dtype=torch.bfloat16))
    tables = torch.tensor(tables)
    q = torch.randn(3, 8, D, dtype=torch.float32)
    k3, v3 = cache.gather_fixed(tables, torch.float32)
    out3 = decode_attention_fixed(q, k3, v3, torch.tensor(lens), 2, D ** -0.5)
    k1, v1 = cache.gather_fixed(tables[:1], torch.float32)
    out1 = decode_attention_fixed(q[:1], k1, v1, torch.tensor(lens[:1]), 2, D ** -0.5)
    assert torch.equal(out3[:1], out1)


def test_budget_octets():
    assert kv_k8v4.octets_par_jeton_couche(HKV, D) == 808           # contre 1 040 en int8
    cfg = _cfg()
    assert cfg.bytes_per_block() == 16 * 808
    assert cfg.total_bytes() == 8 * 16 * 808
    assert PagedKVCache(cfg).nbytes == cfg.total_bytes()
    from acvram.engine.config import ModelSpec

    class _Coder:                                                    # duck-typé : Coder 30B-A3B
        est_mla, num_key_value_heads, head_dim, couches_avec_kv = False, 4, 128, 48
    assert ModelSpec.kv_bytes_per_token(_Coder(), 8, fmt="k8v4") == 38_784
    assert ModelSpec.kv_bytes_per_token(_Coder(), 8) == 49_920


def test_config_k8v4_formes_et_canal_refuse():
    cfg = _cfg(canal=True)
    assert cfg.k8v4 and cfg.quantized and not cfg.rotated
    assert cfg.canal is False                      # C5-b est un chemin int8 seulement
    assert cfg.nom_format == "k8v4"                # ligne de régime kv=k8v4
    assert cfg.torch_dtype == torch.int8 and cfg.torch_dtype_v == torch.uint8
    assert cfg.storage_dim == D and cfg.storage_dim_v == D // 2 and cfg.echelles_v == (4,)
    with pytest.raises(ValueError):
        kv_k8v4.groupes(48)


def test_variable_regime_documentee():
    from acvram import regime
    var = next(v for v in regime.VARIABLES if v.nom == "KV_FORMAT")
    assert re.search(r"k8v4", var.note)


def test_graphes_paged_ok_admet_k8v4_et_le_repli_est_nomme():
    """Revue poste1 (p104 v1) : `paged_ok` exigeait dtype == "int8" — en k8v4 la
    vérification spéculative (q_len > 1) ne se capturait jamais, et `run()`
    rendait False sans `_eager` nommé. Deux gardes : la source du prédicat porte
    k8v4 (casse si l'on revient à "int8" seul) ; un lot spéculatif sans noyau
    paginé est COMPTÉ et NOMMÉ par `_eager`."""
    import inspect
    from acvram.engine import graphs
    src = inspect.getsource(graphs.GraphRunner._eligible)
    assert 'cfg.dtype in ("int8", "k8v4")' in src

    class Lot:
        is_prefill = False
        query_lens = [4, 4]
        batch_size = 2
    g = graphs.GraphRunner.__new__(graphs.GraphRunner)
    g.enabled, g.paged_ok, g._raisons_eager_vues = True, False, set()
    assert g.preparer(Lot()) is False
    assert g.replis_eager == 1
    assert any("spéculative" in r for r in g._raisons_eager_vues)
