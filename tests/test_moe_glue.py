"""Glue du prefill MoE en deux noyaux (moe_act, moe_reduce_trie) contre la
version torch qu'ils remplacent, sur les mêmes entrées.

Pas bit-identique par construction : l'extension est compilée avec
--use_fast_math (expf/tanhf approchés) et torch réduit dans son ordre. La
tolérance est un ulp bf16 sur la sortie, et la fraction identique est
rapportée."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "moe_act") or not hasattr(ext, "moe_reduce_trie"):
        pytest.skip("extension sans la glue MoE")
    return ext


def _proche(a, b, part_min=0.99):
    a, b = a.float(), b.float()
    tol = b.abs() * 2 ** -7 + 1e-6
    hors = int(((a - b).abs() > tol).sum())
    ident = (a == b).float().mean().item()
    assert hors == 0, f"{hors} valeurs hors tolérance sur {a.numel()}, max {(a - b).abs().max().item():.3e}"
    assert ident >= part_min, f"seulement {ident:.4f} identiques"


@pytest.mark.parametrize("G,Mp,m,Kd", [(37, 768, 768, 768), (200, 832, 768, 768), (5, 1600, 1500, 1536), (64, 64, 64, 128)])
@pytest.mark.parametrize("act", [0, 1])
def test_moe_act(G, Mp, m, Kd, act):
    ext = _ext()
    g = torch.Generator(device="cuda").manual_seed(G * 7 + m)
    gg = (torch.randn(G, Mp, device="cuda", generator=g) * 3).to(torch.bfloat16)
    uu = (torch.randn(G, Mp, device="cuda", generator=g) * 2).to(torch.bfloat16)
    y = ext.moe_act(gg, uu, m, Kd, act)
    assert y.shape == (G, Kd) and y.dtype == torch.bfloat16
    a = F.gelu(gg[:, :m].float(), approximate="tanh") if act else F.silu(gg[:, :m].float())
    ref = (a * uu[:, :m].float()).to(torch.bfloat16)
    ref = F.pad(ref, (0, Kd - m))
    _proche(y, ref)
    if Kd > m:
        assert torch.equal(y[:, m:], torch.zeros_like(y[:, m:]))


@pytest.mark.parametrize("t,k,Mp,m", [(1, 8, 2048, 2048), (17, 8, 2048, 2048), (256, 8, 2112, 2048), (33, 4, 256, 200)])
def test_moe_reduce_trie(t, k, Mp, m):
    ext = _ext()
    G = t * k
    g = torch.Generator(device="cuda").manual_seed(t * 13 + k)
    d = (torch.randn(G, Mp, device="cuda", generator=g)).to(torch.bfloat16)
    topw = torch.rand(t, k, device="cuda", generator=g)
    topw = (topw / topw.sum(-1, keepdim=True)).float()
    ordre = torch.randperm(G, device="cuda", generator=g)
    inv = torch.empty_like(ordre); inv[ordre] = torch.arange(G, device="cuda")
    y = ext.moe_reduce_trie(d, topw.reshape(-1).contiguous(), inv.to(torch.int32).contiguous(), m, k)
    assert y.shape == (t, m) and y.dtype == torch.bfloat16
    # la version torch remplacée, mot pour mot
    dd = d[:, :m].to(torch.float32) * topw.reshape(-1)[ordre].unsqueeze(-1)
    ref = dd[inv].view(t, k, -1).sum(dim=1).to(torch.bfloat16)
    _proche(y, ref, part_min=0.98)


def test_moe_reduce_trie_permutation_lue():
    """Un contrôle qui peut rendre faux : inv permuté change la sortie."""
    ext = _ext()
    t, k, m = 9, 8, 256
    G = t * k
    d = torch.randn(G, m, device="cuda").to(torch.bfloat16)
    topw = torch.ones(G, device="cuda") / k
    ident = torch.arange(G, device="cuda", dtype=torch.int32)
    y1 = ext.moe_reduce_trie(d, topw, ident, m, k)
    perm = torch.randperm(G, device="cuda").to(torch.int32)
    y2 = ext.moe_reduce_trie(d, topw, perm, m, k)
    assert not torch.equal(y1, y2)
