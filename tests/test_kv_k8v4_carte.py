"""Pièce 104 sur carte : `kv_write_k8v4` et `paged_attention_k8v4` contre le
jumeau torch (memory/kv_k8v4.py) et la référence fp32. Sauté sans CUDA ou
sans les symboles ; à lancer sous `outils/carte.sh` seulement.

* écriture : K (int8 par jeton) et V (quartets + échelles par groupe) AU BIT
  contre le jumeau, demi-entiers exacts compris (rn pair) ; sentinelle ;
* lecture : ≤ 2⁻⁸ du max de la ligne contre l'attention fp32 sur le cache
  déquantifié, frontières de bloc et de tranche (513, 8 193), fenêtre ;
* lot mêlé Δ = 0 : la même séquence seule ou dans un lot, au bit ;
* reproductible 20 × ; graphe CUDA capturé et rejoué = eager au bit.
"""
from __future__ import annotations

import math

import pytest
import torch

from acvram.memory import kv_canal, kv_k8v4
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS, N_REP = 4, 128, 16, 8
TOL = 2 ** -8


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_k8v4") or not hasattr(ext, "paged_attention_k8v4"):
        pytest.skip("extension absente ou sans les symboles k8v4")
    return ext


def _cache(num_blocks):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=num_blocks,
                                      dtype="k8v4", device="cuda"))


def _kv(n, graine):
    torch.manual_seed(graine)
    k = torch.randn(n, HKV, D, device="cuda").to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device="cuda").to(torch.bfloat16)
    return k, v


def _ecrire_jumeau(c, slots, k, v):
    kq, ks = c._quantize(k)
    vq, vs = kv_k8v4.quantifier_v(v)
    blk, off = torch.div(slots, BS, rounding_mode="floor"), slots % BS
    c.k[blk, off] = kq
    c.v[blk, off] = vq
    c.k_scale[blk, off] = ks
    c.v_scale[blk, off] = vs


def _etat(c):
    return (c.k.clone(), c.v.clone(), c.k_scale.clone(), c.v_scale.clone())


def _memes(a, b, quoi):
    for x, y, nom in zip(a, b, ("k", "v", "k_scale", "v_scale")):
        assert torch.equal(x, y), f"{quoi} : {nom} diffère du jumeau"


def test_ecriture_au_bit_contre_le_jumeau_demi_entiers_compris():
    ext = _ext()
    k, v = _kv(37, 104)
    # demi-entiers exacts : groupe d'amax 7 → sv = 1 ; 3,5 → 4 (pair), 2,5 → 2, −0,5 → 0
    v[3, 1, :32] = 0.0
    v[3, 1, 0], v[3, 1, 1], v[3, 1, 2], v[3, 1, 3] = 7.0, 3.5, 2.5, -0.5
    torch.manual_seed(3)
    slots = torch.randperm(6 * BS)[:37].to("cuda")                 # dispersés, tous DISTINCTS
    c_n, c_j = _cache(6), _cache(6)
    c_n.write(slots, k, v)                       # noyau (dispatch k8v4 de PagedKVCache.write)
    _ecrire_jumeau(c_j, slots, k, v)
    _memes(_etat(c_n), _etat(c_j), "écriture")
    s3 = int(slots[3])                                   # la ligne 3 est allée au slot s3
    codes = kv_k8v4.deballer(c_n.v[s3 // BS, s3 % BS, 1:2].cpu())
    assert codes[0, :4].tolist() == [7, 4, 2, 0]


def test_sentinelle_slot_negatif_n_ecrit_rien():
    _ext()
    c = _cache(2)
    avant = _etat(c)
    k, v = _kv(3, 105)
    c.write(torch.tensor([-1, -1, -1], device="cuda"), k, v)
    _memes(_etat(c), avant, "sentinelle")


def _montage(lens, graine=0):
    B = len(lens)
    nb = max(1, (max(lens) + BS - 1) // BS) + 1     # un bloc de marge : le graphe écrit 16 jetons de plus
    c = _cache(B * nb + 1)
    tables = torch.arange(1, B * nb + 1, device="cuda").view(B, nb)
    for b, n in enumerate(lens):
        if n:
            slots = torch.tensor([int(tables[b, i // BS]) * BS + i % BS for i in range(n)], device="cuda")
            k, v = _kv(n, graine + b)
            c.write(slots, k, v)
    torch.manual_seed(graine + 100)
    q = torch.randn(B, HKV * N_REP, D, device="cuda").to(torch.bfloat16)
    return c, tables, torch.tensor(lens, device="cuda"), q


def _attention(ext, c, q, tables, lens, scale, window=0, q_len=1):
    return ext.paged_attention_k8v4(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale,
                                    tables.contiguous(), lens.contiguous(), HKV, float(scale),
                                    int(q_len), int(window))


def _reference(c, tables, lens, q, scale, window=0):
    B, HQ, _ = q.shape
    ref = torch.zeros(B, HQ, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        n = int(lens[b])
        if n:
            k, v = c.gather(tables[b], n, torch.float32)
            ref[b] = kv_canal.attention_reference(q[b], k, v, scale, window)
    return ref


def _hors(out, ref, tol=TOL):
    ecart = (out.float() - ref).abs().amax(-1)
    borne = ref.abs().amax(-1).clamp(min=1e-6) * tol
    return int((ecart > borne).sum()), float((ecart / borne).max())


@pytest.mark.parametrize("lens", [[37, 0, 300], [1], [2048, 17], [16, 32, 33, 1], [513, 8193]])
def test_attention_k8v4_suit_la_reference_fp32_a_2_moins_8(lens):
    ext = _ext()
    c, tables, L, q = _montage(lens)
    scale = 1 / math.sqrt(D)
    for window in (0, 100):
        out = _attention(ext, c, q, tables, L, scale, window)
        assert torch.isfinite(out.float()).all()
        hors, pire = _hors(out, _reference(c, tables, L, q, scale, window))
        assert hors == 0, f"fenêtre {window} : {hors} lignes hors 2^-8, pire {pire:.2f} × la tolérance"
        for b, n in enumerate(lens):
            if n == 0:
                assert not out[b].float().any(), "un fantôme du godet doit rendre zéro"


def test_q_len_4_verification_speculative_suit_la_reference():
    """q_len = 4 : la requête qi d'une séquence de longueur n ne voit que les
    n − 3 + qi premières positions (causalité) ; référence fp32 par qi."""
    ext = _ext()
    QL = 4
    lens = [40, 7, 300]
    c, tables, L, _ = _montage(lens, graine=11)
    B = len(lens)
    torch.manual_seed(12)
    q = torch.randn(B * QL, HKV * N_REP, D, device="cuda").to(torch.bfloat16)
    scale = 1 / math.sqrt(D)
    for window in (0, 20):
        out = _attention(ext, c, q, tables, L, scale, window, q_len=QL)
        ref = torch.zeros_like(out, dtype=torch.float32)
        for b, n in enumerate(lens):
            k, v = c.gather(tables[b], n, torch.float32)
            for qi in range(QL):
                m = n - (QL - 1) + qi
                ref[b * QL + qi] = kv_canal.attention_reference(q[b * QL + qi], k[:m], v[:m], scale, window)
        hors, pire = _hors(out, ref)
        assert hors == 0, f"q_len 4, fenêtre {window} : {hors} lignes hors 2^-8, pire {pire:.2f}"


def test_lot_mele_delta_zero_et_reproductible_20x():
    ext = _ext()
    c, tables, L, q = _montage([37, 5, 21], graine=7)
    scale = 1 / math.sqrt(D)
    out3 = _attention(ext, c, q, tables, L, scale)
    out1 = _attention(ext, c, q[:1], tables[:1], L[:1], scale)
    assert torch.equal(out3[:1], out1), "la même séquence rend autre chose dans un lot : fuite d'adresse"
    for _ in range(20):
        assert torch.equal(_attention(ext, c, q, tables, L, scale), out3)


def test_graphe_cuda_ecriture_et_attention_rejouees_egalent_l_eager():
    ext = _ext()
    c_g, tables, L, q = _montage([40], graine=40)
    c_e, _, _, _ = _montage([40], graine=40)
    scale = 1 / math.sqrt(D)
    slot_s = torch.zeros(1, dtype=torch.long, device="cuda")
    k_s = torch.zeros(1, HKV, D, dtype=torch.bfloat16, device="cuda")
    v_s = torch.zeros(1, HKV, D, dtype=torch.bfloat16, device="cuda")
    len_s = torch.zeros(1, dtype=torch.long, device="cuda")

    def pas(c):
        c.write(slot_s, k_s, v_s)
        return _attention(ext, c, q, tables, len_s, scale)

    ks, vs = _kv(16, 41)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):                    # chauffe hors capture sur un 3e cache
        c_w, _, _, _ = _montage([40], graine=40)
        slot_s.fill_(int(tables[0, 2]) * BS + 8); k_s.copy_(ks[:1]); v_s.copy_(vs[:1]); len_s.fill_(41)
        pas(c_w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = pas(c_g)
    for j in range(16):
        pos = 40 + j
        slot_s.fill_(int(tables[0, pos // BS]) * BS + pos % BS)
        k_s.copy_(ks[j:j + 1]); v_s.copy_(vs[j:j + 1]); len_s.fill_(pos + 1)
        g.replay()
        out_e = pas(c_e)
        torch.cuda.synchronize()
        assert torch.equal(out_s, out_e), f"pas {pos}"
    _memes(_etat(c_g), _etat(c_e), "graphe")
