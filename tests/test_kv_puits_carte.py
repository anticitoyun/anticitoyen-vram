"""Repli 104 (1) sur carte : `kv_write_k8v4_puits` et `paged_attention_k8v4_puits` contre le jumeau torch
(`test_kv_puits.py`) et la référence fp32. Sauté sans CUDA ou sans les symboles ; sous `outils/carte.sh` seulement.

* écriture : K, quartets, échelles ET réserve des puits AU BIT contre le jumeau (positions mêlées, puits et non puits) ;
* lecture : ≤ 2⁻⁸ du max de la ligne contre l'attention fp32 sur le cache relu (puits superposés), séquences plus
  courtes et plus longues que 16 ;
* la réserve est LUE : la sortie diffère de `paged_attention_k8v4` (sans puits) dès qu'une position puits pèse ;
* graphe CUDA capturé et rejoué = eager au bit (écriture + attention).
"""
from __future__ import annotations

import math

import pytest
import torch

from acvram.memory import kv_k8v4
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS, N_REP = 4, 128, 16, 8
TOL = 2 ** -8


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_k8v4_puits") or not hasattr(ext, "paged_attention_k8v4_puits"):
        pytest.skip("extension absente ou sans les symboles des puits")
    return ext


def _cache(nb):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=nb,
                                      dtype="k8v4", device="cuda", puits=16))


def _jumeau(c, slots, k, v, positions):
    """Le chemin torch de `PagedKVCache.write`, appelé explicitement (le noyau est court-circuité)."""
    kq, ks = kv_k8v4.quantifier_k(k)
    vq, vs = kv_k8v4.quantifier_v(v)
    blk, off = torch.div(slots, BS, rounding_mode="floor"), slots % BS
    c.k[blk, off], c.v[blk, off], c.k_scale[blk, off], c.v_scale[blk, off] = kq, vq, ks, vs
    p = positions < 16
    pq, ps = kv_k8v4.quantifier_k(v[p])
    c.puits_v[blk[p], off[p]], c.puits_vs[blk[p], off[p]] = pq, ps


def _kv(n, graine):
    torch.manual_seed(graine)
    return (torch.randn(n, HKV, D, device="cuda").to(torch.bfloat16),
            torch.randn(n, HKV, D, device="cuda").to(torch.bfloat16))


def test_ecriture_au_bit_contre_le_jumeau_reserve_comprise():
    _ext()
    k, v = _kv(40, 1)
    positions = torch.tensor(list(range(12)) + list(range(10, 38)), device="cuda")   # deux séquences mêlées
    blocs = ([3], [6, 4, 5])                                   # séquence 1 : 12 jetons ; séquence 2 : 10..37
    slots = torch.tensor([blocs[int(i >= 12)][int(p) // BS] * BS + int(p) % BS for i, p in enumerate(positions.tolist())],
                         device="cuda")
    a, b = _cache(8), _cache(8)
    a.write(slots, k, v, positions=positions)
    _jumeau(b, slots, k, v, positions)
    torch.cuda.synchronize()
    for x, y, nom in ((a.k, b.k, "k"), (a.v, b.v, "v"), (a.k_scale, b.k_scale, "ks"), (a.v_scale, b.v_scale, "vs"),
                      (a.puits_v, b.puits_v, "puits_v"), (a.puits_vs, b.puits_vs, "puits_vs")):
        assert torch.equal(x, y), nom


def _montage(lens, graine=0):
    B = len(lens)
    nb = max(1, (max(lens) + BS - 1) // BS) + 1
    tables = torch.arange(1, B * nb + 1, device="cuda").view(B, nb)
    c = _cache(B * nb + 2)
    for b, n in enumerate(lens):
        k, v = _kv(n, graine + b)
        slots = torch.tensor([int(tables[b, i // BS]) * BS + i % BS for i in range(n)], device="cuda")
        c.write(slots, k, v, positions=torch.arange(n, device="cuda"))
    torch.manual_seed(graine + 99)
    q = torch.randn(B, HKV * N_REP, D, device="cuda").to(torch.bfloat16)
    return c, tables, torch.tensor(lens, device="cuda"), q


def _attention(ext, c, q, tables, lens, scale, puits=True):
    if puits:
        return ext.paged_attention_k8v4_puits(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale, c.puits_v, c.puits_vs,
                                              tables.contiguous(), lens.contiguous(), HKV, float(scale), 1, 0, 16)
    return ext.paged_attention_k8v4(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale,
                                    tables.contiguous(), lens.contiguous(), HKV, float(scale), 1, 0)


def _reference(c, tables, lens, q, scale):
    outs = []
    for b, n in enumerate(lens.tolist()):
        k, v = c.gather(tables[b], n, torch.float32)
        kk = k.repeat_interleave(N_REP, dim=1)
        vv = v.repeat_interleave(N_REP, dim=1)
        s = torch.einsum("hd,thd->ht", q[b].float(), kk) * scale
        outs.append(torch.einsum("ht,thd->hd", s.softmax(-1), vv))
    return torch.stack(outs)


@pytest.mark.parametrize("lens", [[5], [16], [17], [40, 3], [300, 16, 1]])
def test_attention_puits_suit_la_reference_fp32_a_2_moins_8(lens):
    ext = _ext()
    c, tables, L, q = _montage(lens)
    scale = 1 / math.sqrt(D)
    out = _attention(ext, c, q, tables, L, scale).float()
    ref = _reference(c, tables, L, q, scale)
    ecart = (out - ref).abs().amax(-1) / ref.abs().amax(-1).clamp(min=1e-6)
    assert float(ecart.max()) <= TOL, float(ecart.max())


def test_la_reserve_est_lue():
    ext = _ext()
    c, tables, L, q = _montage([12], graine=5)
    scale = 1 / math.sqrt(D)
    assert not torch.equal(_attention(ext, c, q, tables, L, scale), _attention(ext, c, q, tables, L, scale, puits=False))


def test_graphe_cuda_ecriture_et_attention_rejouees_egalent_l_eager():
    ext = _ext()
    c, tables, L, q = _montage([9], graine=40)
    scale = 1 / math.sqrt(D)
    k1, v1 = _kv(1, 77)
    slot_s = torch.tensor([int(tables[0, 0]) * BS + 9], device="cuda")
    pos_s = torch.tensor([9], device="cuda")
    len_s = torch.tensor([10], device="cuda")

    def pas():
        c.write(slot_s, k1, v1, positions=pos_s)
        return _attention(ext, c, q, tables, len_s, scale)

    eager = pas().clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pas()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sortie = pas()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(sortie, eager)
