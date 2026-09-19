"""C15 niveau 3 (revue/chantier-c15-niveau3-coder-20-09), ACVRAM_GLUE_COMPACT :
les trois retraits de glue AU BIT du pas GQA (item 2 de la fiche) —

* `kv_write_int8` lit k et v par leur pas de jeton (tranches de la projection
  q/k/v empilée) au lieu de deux copies contiguës par couche
  (acvram_kernels.cu, `contigu_par_tete`) ; le témoin les recopie en Python
  (kvcache.write) ;
* l'attention paginée Triton lit q par ses pas (`_partiel_kernel`, stride_qb /
  stride_qh) : plus de `q.contiguous()` dans kernels.paged_attention ;
* `valid = slots >= 0` calculé une fois par pas (ACVRamModel.decode_fixed)
  au lieu d'une fois par couche (DecoderLayer.decode_fixed_res).

Sans carte : l'attention Triton sous ``TRITON_INTERPRET=1`` (q fp16) et la
plomberie ; le noyau CUDA (pas de jeton) n'a de preuve que sur carte.
"""
import importlib
import math
import os

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ap():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    ap = importlib.import_module("acvram.kernels.attn_paginee")
    if not ap.disponible():
        pytest.skip("Triton indisponible")
    return ap


def _cache(lens, hkv=2, d=128, graine=0):
    torch.manual_seed(graine)
    B = len(lens)
    N = -(-max(lens) // 16) + 1
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=hkv, head_dim=d,
                                   num_blocks=B * N, dtype="int8", device=DEV))
    tables = torch.arange(B * N, device=DEV).view(B, N)
    for b, n in enumerate(lens):
        slots = torch.tensor([int(tables[b, i // 16]) * 16 + i % 16 for i in range(n)], device=DEV)
        c.write(slots, torch.randn(n, hkv, d, device=DEV), torch.randn(n, hkv, d, device=DEV))
    return c, tables, torch.tensor(lens, device=DEV)


def test_attention_triton_lit_q_par_ses_pas_au_bit():
    """q tranche d'une projection empilée [B, (HQ + 2·HKV)·D] → même sortie,
    au bit, que la copie contiguë ; et le bras qui casse : la tranche voisine
    (k) donnée pour q rend autre chose."""
    ap = _ap()
    c, tables, L = _cache([37, 300])
    B, hkv, n_rep, d = 2, 2, 8, 128
    dt = torch.bfloat16 if DEV == "cuda" else torch.float16
    qkv = torch.randn(B, (hkv * n_rep + 2 * hkv) * d, device=DEV).to(dt)
    q_vue = qkv[:, : hkv * n_rep * d].view(B, hkv * n_rep, d)          # pas (…, 128, 1) : non contigu
    assert not q_vue.is_contiguous() and q_vue.stride(1) == d
    scale = 1 / math.sqrt(d)
    a = ap.paged_attention(q_vue, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    b = ap.paged_attention(q_vue.contiguous(), c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    assert torch.equal(a, b)
    k_vue = qkv[:, d: d + hkv * n_rep * d].view(B, hkv * n_rep, d)  # décalée d'une tête
    autre = ap.paged_attention(k_vue, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    assert not torch.equal(a, autre)


def test_lanceur_ne_recopie_q_que_sous_le_temoin(monkeypatch):
    """kernels.paged_attention : sous GLUE_COMPACT=1 la vue passe telle
    quelle ; sous 0 (témoin) une copie contiguë — les nœuds d'avant."""
    from acvram import kernels
    ap = _ap()
    vus = []
    monkeypatch.setattr(ap, "paged_attention", lambda q, *a, **k: vus.append(q) or torch.zeros_like(q))
    monkeypatch.setattr(kernels, "get_extension", lambda: object())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    c, tables, L = _cache([16, 16])
    q = torch.randn(2, 16 * 128 + 256).to(torch.bfloat16)[:, :16 * 128].view(2, 16, 128)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    kernels.paged_attention(q, c, tables, L, 8, 0.1)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    kernels.paged_attention(q, c, tables, L, 8, 0.1)
    assert vus[0].data_ptr() == q.data_ptr() and not vus[0].is_contiguous()
    assert vus[1].data_ptr() != q.data_ptr() and vus[1].is_contiguous()


def test_kv_write_temoin_recopie_compact_non(monkeypatch):
    """kvcache.write : le témoin donne au noyau deux copies contiguës, le
    compact les tranches telles quelles (le .cu lit `stride(0)`)."""
    from acvram import kernels
    vus = []

    class Ext:
        def kv_write_int8(self, k, v, *a):
            vus.append((k, v))
    monkeypatch.setattr("acvram.kernels.get_extension", lambda: Ext())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=2, head_dim=64, num_blocks=2,
                                   dtype="int8", device="cpu"))
    qkv = torch.randn(3, 8 * 64 + 4 * 64).to(torch.bfloat16)
    k = qkv[:, 8 * 64: 10 * 64].view(3, 2, 64)
    v = qkv[:, 10 * 64:].view(3, 2, 64)
    slots = torch.arange(3)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    c.write(slots, k, v)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    c.write(slots, k, v)
    assert vus[0][0].data_ptr() == k.data_ptr() and vus[0][1].data_ptr() == v.data_ptr()
    assert vus[1][0].is_contiguous() and vus[1][1].is_contiguous() and vus[1][0].data_ptr() != k.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA : carte seulement")
def test_kv_write_int8_par_tranches_au_bit_sur_carte():
    """Sur carte : le cache écrit depuis les tranches (pas de jeton 12·64)
    est identique au bit à celui écrit depuis les copies contiguës, codes et
    échelles ; le bras qui casse : les tranches k et v échangées."""
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8"):
        pytest.skip("extension absente")
    torch.manual_seed(1)
    T, H, D = 12, 4, 64
    qkv = torch.randn(T, (8 + 2 * H) * D, device="cuda").to(torch.bfloat16)
    k = qkv[:, 8 * D: (8 + H) * D].view(T, H, D)
    v = qkv[:, (8 + H) * D:].view(T, H, D)
    slots = torch.arange(T, device="cuda")

    def ecrit(kk, vv):
        c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=H, head_dim=D, num_blocks=2,
                                       dtype="int8", device="cuda"))
        ext.kv_write_int8(kk, vv, slots, c.k.view(-1, *c.k.shape[2:]), c.v.view(-1, *c.v.shape[2:]),
                          c.k_scale.view(-1, c.k_scale.shape[-1]), c.v_scale.view(-1, c.v_scale.shape[-1]), 16)
        return c
    a, b = ecrit(k, v), ecrit(k.contiguous(), v.contiguous())
    assert all(torch.equal(getattr(a, n), getattr(b, n)) for n in ("k", "v", "k_scale", "v_scale"))
    faux = ecrit(v, k)
    assert not torch.equal(faux.k, a.k)


def test_valid_une_fois_par_pas_est_transmis(monkeypatch):
    """DecoderLayer.decode_fixed_res : `valid` reçu est donné tel quel au
    MoE ; sans lui, la couche calcule `slots >= 0` elle-même (témoin)."""
    from acvram.engine.model import DecoderLayer, MoEBlock
    from acvram.engine.layers import RMSNorm
    vus = []

    class Attn:
        def decode_fixed(self, h, *a, **k):
            return h

    class Moe(MoEBlock):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, h, valid=None):
            vus.append(valid)
            return h

    n1, n2 = RMSNorm(torch.ones(8, dtype=torch.bfloat16)), RMSNorm(torch.ones(8, dtype=torch.bfloat16))
    couche = DecoderLayer(0, Attn(), Moe(), n1, n2, torch.device("cpu"))
    x = torch.randn(4, 8).to(torch.bfloat16)
    slots = torch.tensor([0, 1, -1, -1])
    valid = slots >= 0
    couche.decode_fixed_res(x, None, None, slots, None, None, 4, None, valid=valid)
    couche.decode_fixed_res(x, None, None, slots, None, None, 4, None)
    assert vus[0] is valid and vus[1] is not valid and torch.equal(vus[1], valid)
