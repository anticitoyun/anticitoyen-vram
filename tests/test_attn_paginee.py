"""Juge du poste E (kernels/attn_paginee.py) : l'attention de décodage
paginée Triton contre une référence float64 lue du même cache int8, à
± 2⁻⁸ du maximum de chaque ligne (b, tête) — le scellé de Sage —, et cinq
bras qui doivent casser ou différer : table de blocs permutée, échelle d'un
jeton effacée, fenêtre glissante qui change la sortie, groupe GQA permuté,
fantôme du godet nul et fini. Sur carte, en plus : Triton = noyau CUDA
actuel ± 2⁻⁸ sur les mêmes tenseurs.

Sans carte : ``TRITON_INTERPRET=1`` posé avant l'import, q en fp16.
"""
import importlib
import math
import os

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

TOL = 2 ** -8


def _ap():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    ap = importlib.import_module("acvram.kernels.attn_paginee")
    if not ap.disponible():
        pytest.skip("Triton indisponible")
    return ap


def _montage(lens, hkv=2, n_rep=8, d=128, graine=0):
    """Un cache int8 rempli de k/v gaussiens pour `lens` séquences, tables
    disjointes, q fp16 (bf16 sur carte)."""
    torch.manual_seed(graine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = len(lens)
    N = -(-max(max(lens), 1) // 16) + 1
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=hkv, head_dim=d,
                                   num_blocks=B * N, dtype="int8", device=dev))
    tables = torch.arange(B * N, device=dev).view(B, N)
    for b, n in enumerate(lens):
        if n:
            slots = torch.tensor([int(tables[b, i // 16]) * 16 + i % 16 for i in range(n)], device=dev)
            k = torch.randn(n, hkv, d, device=dev)
            k[0] *= 20                                   # puits en position 0
            c.write(slots, k, torch.randn(n, hkv, d, device=dev))
    q = torch.randn(B, hkv * n_rep, d, device=dev).to(torch.bfloat16 if dev == "cuda" else torch.float16)
    return c, tables, torch.tensor(lens, device=dev), q


def _reference(c, tables, lens, q, n_rep, scale, window=0):
    B, HQ, D = q.shape
    ref = torch.zeros(B, HQ, D, dtype=torch.float64, device=q.device)
    for b in range(B):
        n = int(lens[b])
        if n == 0:
            continue
        k, v = c.gather(tables[b], n, torch.float64)
        lo = max(0, n - window) if window else 0
        for h in range(HQ):
            kh, vh = k[lo:, h // n_rep], v[lo:, h // n_rep]
            p = torch.softmax((kh @ q[b, h].double()) * scale, 0)
            ref[b, h] = p @ vh
    return ref


def _hors(out, ref):
    ecart = (out.double() - ref).abs().amax(-1)                  # [B, HQ]
    borne = ref.abs().amax(-1).clamp(min=1e-6) * TOL
    return int((ecart > borne).sum()), float((ecart / borne).max())


@pytest.mark.parametrize("lens", [[37, 0, 300], [1], [2048, 17], [16, 32, 33, 1]])
def test_la_sortie_suit_la_reference_a_2_moins_8(lens):
    ap = _ap()
    c, tables, L, q = _montage(lens)
    scale = 1 / math.sqrt(q.shape[-1])
    out = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, 2, scale)
    assert torch.isfinite(out.float()).all()
    hors, pire = _hors(out, _reference(c, tables, L, q, 8, scale))
    assert hors == 0, f"{hors} lignes hors 2^-8, pire {pire:.2f} × la tolérance"
    for b, n in enumerate(lens):
        if n == 0:
            assert not out[b].float().any(), "un fantôme du godet doit rendre zéro"


def test_la_fenetre_glissante_change_la_sortie_et_suit_la_reference():
    ap = _ap()
    c, tables, L, q = _montage([300, 37])
    scale = 1 / math.sqrt(128)
    sans = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, 2, scale, 0)
    avec = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, 2, scale, 100)
    assert not torch.allclose(sans[0].float(), avec[0].float(), atol=1e-3), "la fenêtre ne change rien à 300 jetons"
    assert torch.equal(sans[1], avec[1]), "37 < 100 : la fenêtre ne doit rien changer"
    hors, pire = _hors(avec, _reference(c, tables, L, q, 8, scale, window=100))
    assert hors == 0, (hors, pire)


def test_les_bras_qui_doivent_casser():
    """Table permutée, échelle effacée, groupe GQA permuté : chacun DOIT sortir
    de tolérance — sinon le juge ne lit pas ce qu'il croit lire."""
    ap = _ap()
    c, tables, L, q = _montage([300, 37])
    scale = 1 / math.sqrt(128)
    ref = _reference(c, tables, L, q, 8, scale)
    f = lambda **kw: ap.paged_attention(kw.get("q", q), c.k, kw.get("ks", c.k_scale), c.v, c.v_scale,
                                       kw.get("tables", tables), L, 2, scale)
    t2 = tables.clone(); t2[0] = t2[0].roll(1)
    assert _hors(f(tables=t2), ref)[0] > 0, "table permutée invisible"
    ks2 = c.k_scale.clone(); ks2[int(tables[0, 0]), 0] = 0        # le puits perd son échelle
    assert _hors(f(ks=ks2), ref)[0] > 0, "échelle effacée invisible"
    q2 = q.view(2, 2, 8, 128).flip(1).reshape(2, 16, 128).contiguous()   # têtes des deux groupes échangées
    assert _hors(f(q=q2), ref)[0] > 0, "groupe GQA permuté invisible"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_triton_egale_le_noyau_cuda_actuel():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None:
        pytest.skip("extension absente")
    ap = _ap()
    for lens in ([37, 0, 300], [2048, 17], [1]):
        c, tables, L, q = _montage(lens)
        scale = 1 / math.sqrt(128)
        for window in (0, 100):
            tri = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, 2, scale, window)
            cuda = ext.paged_attention(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale, tables.contiguous(),
                                       L.contiguous(), 2, float(scale), 1, int(window))
            vivants = [b for b, n in enumerate(lens) if n]
            # Chacun contre la référence fp64, pas l'un contre l'autre : deux noyaux à
            # 0,94 et 0,89 × la borne de part et d'autre de la référence sont à 1,83 ×
            # entre eux et tous deux conformes (Manon, poste E, 20/09 : [37, 0, 300],
            # séquence 0, tête 15 — le test se cassait par construction).
            ref = _reference(c, tables, L, q, 8, scale, window=window)[vivants]   # n_rep = 16 têtes q / 2 kv
            for nom, out in (("triton", tri), ("cuda", cuda)):
                hors, pire = _hors(out[vivants], ref)
                assert hors == 0, (nom, lens, window, hors, pire)
            hors, pire = _hors(tri[vivants], cuda[vivants].double())
            assert pire <= 2.0, (lens, window, hors, pire)     # entre eux : au plus 2 × TOL
