"""anticitoyen-vram-thf, sur carte : `kv_write_int8_kernel` (acvram_kernels.cu:4190, chemin par
jeton servi par défaut pour K et V) et `kvc_quant_par_jeton` (:4415, sa copie pour V du chemin
canal C5-b) divergeaient du jumeau torch (`kvcache.py::PagedKVCache._quantize`, `x / scale`) :
sous `--use_fast_math`, `m / 127.f` puis `x * (1.f / sc)` sont des divisions APPROCHÉES, alors
que le jumeau reste en division IEEE correctement arrondie. Correctif : `__fdiv_rn` partout,
division directe (jamais un inverse précalculé multiplié) — comme `kv_write_k8v4_kernel`
(pièce 104) et `kvc_quant_bloc_canal` (déjà juste) le faisaient déjà.

Sauté sans CUDA ou sans extension. À lancer NU (jamais sous un verrou tenu par un autre)."""
from __future__ import annotations

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS = 4, 128, 16


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8"):
        pytest.skip("extension absente ou sans kv_write_int8")
    return ext


def _kv(n, graine):
    torch.manual_seed(graine)
    k = torch.randn(n, HKV, D, device="cuda") * torch.logspace(-1, 1, D, device="cuda")
    v = torch.randn(n, HKV, D, device="cuda") * torch.logspace(-1, 1, D, device="cuda")
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def _cache():
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1,
                                      block_size=BS, dtype="int8", device="cuda", canal=False))


def test_ecriture_int8_par_jeton_au_bit_contre_le_jumeau():
    """Le cassant : sur l'ancien code (division approchée), au moins un code ou une échelle
    diffère du jumeau sur un tirage assez large — après le correctif, zéro écart."""
    _ext()
    cache = _cache()
    k, v = _kv(BS, graine=1)
    slots = torch.arange(BS, device="cuda", dtype=torch.int64)
    cache.write(slots, k, v)

    qk_ref, sk_ref = cache._quantize(k)
    qv_ref, sv_ref = cache._quantize(v)

    qk_obs = cache.k.view(-1, HKV, D)[:BS]
    qv_obs = cache.v.view(-1, HKV, D)[:BS]
    sk_obs = cache.k_scale.view(-1, HKV)[:BS]
    sv_obs = cache.v_scale.view(-1, HKV)[:BS]

    ecarts_k = (qk_obs.int() - qk_ref.int()).abs()
    ecarts_v = (qv_obs.int() - qv_ref.int()).abs()
    assert ecarts_k.max() == 0, f"codes K hors du bit : max |Δ| = {int(ecarts_k.max())}"
    assert ecarts_v.max() == 0, f"codes V hors du bit : max |Δ| = {int(ecarts_v.max())}"
    assert torch.equal(sk_obs, sk_ref), "échelle K hors du bit"
    assert torch.equal(sv_obs, sv_ref), "échelle V hors du bit"


def test_plusieurs_tirages_toujours_au_bit():
    """Plusieurs graines et tailles : la divergence est une question d'arrondi, elle ne se
    voit pas forcément sur un seul tirage — assez d'échantillons pour la faire apparaître si
    le correctif n'était pas là (chaque test seul suffit déjà à casser sur l'ancien code)."""
    _ext()
    for graine in range(5):
        cache = _cache()
        k, v = _kv(BS, graine=100 + graine)
        slots = torch.arange(BS, device="cuda", dtype=torch.int64)
        cache.write(slots, k, v)
        qk_ref, sk_ref = cache._quantize(k)
        qv_ref, sv_ref = cache._quantize(v)
        assert torch.equal(cache.k.view(-1, HKV, D)[:BS], qk_ref), graine
        assert torch.equal(cache.v.view(-1, HKV, D)[:BS], qv_ref), graine
        assert torch.equal(cache.k_scale.view(-1, HKV)[:BS], sk_ref), graine
        assert torch.equal(cache.v_scale.view(-1, HKV)[:BS], sv_ref), graine
