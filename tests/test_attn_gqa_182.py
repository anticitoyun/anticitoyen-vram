"""Pièce 182 (2) : `paged_attn_partial_gqa_kernel` (un bloc par (b·qi, tête kv, tranche), K/V d'un jeton lus une fois
pour les G têtes du groupe) rend les MÊMES OCTETS que `paged_attn_partial_kernel` (un bloc par tête q) : mêmes
tranches, même attribution des jetons aux warps, même arithmétique par tête, même réduction. Routage : le chemin servi
(`kernels.paged_attention`, cache int8, D = 256, G = 6 comme Qwen3.8) doit lancer la variante ; ACVRAM_PA_GQA=0 la
coupe. Témoin : changer les bornes de tranche (ACVRAM_PA_CHUNK) doit changer les octets, sinon l'égalité est aveugle."""
import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _ext():
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "paged_attn_gqa_lancements"):
        pytest.skip("extension sans la variante 182")
    return ext


def _montage(lens, hkv=4, n_rep=6, d=256, q_len=1, graine=0):
    torch.manual_seed(graine)
    B = len(lens)
    N = -(-max(max(lens), 1) // 16) + 1
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=hkv, head_dim=d,
                                   num_blocks=B * N, dtype="int8", device="cuda"))
    tables = torch.arange(B * N, device="cuda").view(B, N)
    for b, n in enumerate(lens):
        if n:
            slots = torch.tensor([int(tables[b, i // 16]) * 16 + i % 16 for i in range(n)], device="cuda")
            k = torch.randn(n, hkv, d, device="cuda")
            k[0] *= 20
            c.write(slots, k, torch.randn(n, hkv, d, device="cuda"))
    q = torch.randn(B * q_len, hkv * n_rep, d, device="cuda").to(torch.bfloat16)
    return c, tables, torch.tensor(lens, device="cuda"), q


def _appel(ext, c, tables, lens, q, hkv, q_len=1, window=0):
    return ext.paged_attention(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale, tables.contiguous(),
                               lens.contiguous(), hkv, q.shape[-1] ** -0.5, q_len, window)


@carte
@pytest.mark.parametrize("n_rep,d", [(6, 256), (4, 128), (8, 128), (6, 64)])
@pytest.mark.parametrize("lens", [[1, 17, 40], [300, 600, 45, 2000, 1, 128, 999, 64]])
@pytest.mark.parametrize("q_len,window", [(1, 0), (1, 100), (3, 0)])
def test_octets_egaux_au_noyau_d_origine(lens, n_rep, d, q_len, window, monkeypatch):
    ext = _ext()
    # seq_len compte les q_len jetons vérifiés, donc seq_len >= q_len en service. En dessous, les lignes à slen <= 0
    # ne sont écrites par AUCUN des deux noyaux (sortie torch::empty) : comparer leurs octets compare du non-initialisé
    # (182, diag-lignes-slen0.log : 12 284 écarts, tous sur ces lignes, 0 sur les lignes valides).
    lens = [max(n, q_len) for n in lens]
    c, tables, L, q = _montage(lens, n_rep=n_rep, d=d, q_len=q_len)
    monkeypatch.setenv("ACVRAM_PA_GQA", "0")
    ref = _appel(ext, c, tables, L, q, 4, q_len, window)
    avant = ext.paged_attn_gqa_lancements()
    monkeypatch.setenv("ACVRAM_PA_GQA", "1")
    out = _appel(ext, c, tables, L, q, 4, q_len, window)
    assert ext.paged_attn_gqa_lancements() == avant + 1, "montage : la variante groupée n'a pas été lancée"
    assert torch.equal(out, ref), f"{int((out != ref).sum())} valeurs diffèrent"


@carte
def test_temoin_les_bornes_de_tranche_changent_les_octets(monkeypatch):
    ext = _ext()
    c, tables, L, q = _montage([300, 600, 2000, 999])
    ref = _appel(ext, c, tables, L, q, 4)
    monkeypatch.setenv("ACVRAM_PA_CHUNK", "128")
    assert not torch.equal(_appel(ext, c, tables, L, q, 4), ref), "l'égalité au bit ne voit pas les tranches"


@carte
def test_le_chemin_servi_lance_la_variante(monkeypatch):
    from acvram import kernels
    ext = _ext()
    c, tables, L, q = _montage([600, 2000, 17])
    monkeypatch.delenv("ACVRAM_PA_GQA", raising=False)
    avant = ext.paged_attn_gqa_lancements()
    out = kernels.paged_attention(q, c, tables, L, 6, 256 ** -0.5)
    assert out is not None and ext.paged_attn_gqa_lancements() == avant + 1, "le chemin servi ne passe pas par 182"
    monkeypatch.setenv("ACVRAM_PA_GQA", "0")
    avant = ext.paged_attn_gqa_lancements()
    temoin = kernels.paged_attention(q, c, tables, L, 6, 256 ** -0.5)
    assert ext.paged_attn_gqa_lancements() == avant and torch.equal(out, temoin)


@carte
@pytest.mark.parametrize("forme", [dict(dtype="bf16"), dict(dtype="k8v4"), dict(dtype="int8", canal=True, rangs=8)])
def test_les_autres_caches_ne_sont_jamais_routes(forme, monkeypatch):
    """KV bf16 (repli, pas de noyau paginé), k8v4 (variante V4) et canal (C5-b) : la variante groupée ne lit que l'int8
    simple ; aucun de ces caches ne doit la lancer, même au défaut."""
    from acvram import kernels
    ext = _ext()
    monkeypatch.delenv("ACVRAM_PA_GQA", raising=False)
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=4, head_dim=256, num_blocks=16, device="cuda", **forme))
    tables = torch.arange(16, device="cuda").view(2, 8)
    L = torch.tensor([40, 90], device="cuda")
    q = torch.randn(2, 24, 256, device="cuda").to(torch.bfloat16)
    avant = ext.paged_attn_gqa_lancements()
    out = kernels.paged_attention(q, c, tables, L, 6, 256 ** -0.5)
    assert ext.paged_attn_gqa_lancements() == avant, f"{forme} routé vers la variante GQA"
    if forme["dtype"] != "bf16":
        assert out is not None, f"montage : {forme} n'a lancé aucun noyau paginé, le test ne prouve rien"
