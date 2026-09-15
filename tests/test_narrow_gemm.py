"""GEMM étroit (M ≤ 16) sur tensor cores bf16 (`narrow_gemm`), poids int8 par
groupes ou NVFP4 par blocs, déquantifiés en registres — 1aj marche 2 élargie.
Contre le GEMV existant (même arithmétique, autre ordre de sommes) à la
tolérance bf16, contre float64, et déterministe (jumelles). M = 1, 12, 16."""
import pytest
import torch

from acvram.quant.formats import quantize

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "narrow_gemm"):
        pytest.skip("extension sans narrow_gemm")
    return ext


def _x(M, K, graine):
    g = torch.Generator(device="cuda").manual_seed(graine)
    return (torch.randn(M, K, device="cuda", generator=g) * 0.5).to(torch.bfloat16).contiguous()


@CUDA
@pytest.mark.parametrize("M,N,K,rows", [(12, 5120, 2048, 32), (1, 2048, 4096, 32), (16, 4096, 2048, 128), (12, 151936 // 8, 2048, 128)])
def test_int8_contre_gemv_et_float64(M, N, K, rows):
    ext = _ext()
    g = torch.Generator().manual_seed(N + K)
    w = torch.randn(N, K, generator=g) * 0.02
    t = quantize(w, "int8", group_size=128)
    qw, sc, zr = (a.cuda().contiguous() for a in (t.qweight, t.scales, t.zeros))
    x = _x(M, K, 7)
    y = ext.narrow_gemm(qw, None, sc, zr, x, K, t.group_size, 1.0, rows)
    y2 = ext.narrow_gemm(qw, None, sc, zr, x, K, t.group_size, 1.0, rows)
    assert torch.equal(y, y2)
    ref = ext.int8_gemv(qw, sc, zr, x, t.group_size)
    # référence float64 depuis les mêmes poids déquantifiés
    wd = ((qw.double() - zr.double().repeat_interleave(t.group_size, 1)) * sc.double().repeat_interleave(t.group_size, 1))
    r64 = x.double() @ wd.T
    tol = r64.abs() * 2 ** -7 + 2e-3 * r64.abs().max()
    assert int(((y.double() - r64).abs() > tol).sum()) == 0
    exact = (y == ref).float().mean().item()
    assert exact > 0.9, f"{exact:.3f} bit-identiques au GEMV int8"


@CUDA
@pytest.mark.parametrize("M,N,K,rows", [(12, 5120, 2048, 32), (1, 2048, 4096, 32), (16, 4096, 2048, 128)])
def test_nvfp4_contre_gemv_et_float64(M, N, K, rows):
    from acvram.quant.nvfp4 import quantize_nvfp4
    from tests.test_gemm_grouped_mma import _w64
    ext = _ext()
    g = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    t = quantize_nvfp4(w)
    qw = t.qweight.contiguous(); bs = t.block_scale.view(torch.uint8).contiguous()
    gs = t.global_scale_float()
    x = _x(M, K, 11)
    y = ext.narrow_gemm(qw, bs, None, None, x, K, 16, gs, rows)
    y2 = ext.narrow_gemm(qw, bs, None, None, x, K, 16, gs, rows)
    assert torch.equal(y, y2)
    ref = ext.nvfp4_gemv(qw, bs, gs, x, K, None)
    wd = _w64(qw.unsqueeze(0), bs.unsqueeze(0)).reshape(N, K) * gs
    r64 = x.double() @ wd.T
    tol = r64.abs() * 2 ** -7 + 2e-3 * r64.abs().max()
    assert int(((y.double() - r64).abs() > tol).sum()) == 0
    exact = (y == ref.to(y.dtype)).float().mean().item()
    assert exact > 0.9, f"{exact:.3f} bit-identiques au GEMV nvfp4"


def test_projections_mla_marquees_etroites(monkeypatch):
    """poste7-duel-verdict § 14 (i) : les projections int8 q/kv/o de l'attention
    MLA portent `etroit` (GEMM étroit à M ≤ 16) après fuse_projections ;
    ACVRAM_NARROW_MLA=0 ne marque rien (témoin GEMV)."""
    import torch.nn as nn
    from acvram.engine.layers import QuantLinear
    from acvram.engine.mla import MLAttention
    from acvram.quant.formats import quantize
    nh, nope, rope, rank, dv, hidden = 4, 32, 16, 64, 32, 256

    def lin(o, i):
        t = quantize(torch.randn(o, i) * 0.05, "int8", group_size=128)
        return QuantLinear(t, out_features=o, in_features=i)
    for env, attendu in (("1", True), ("0", False)):
        monkeypatch.setenv("ACVRAM_NARROW_MLA", env)
        la = MLAttention(lin(nh * (nope + rope), hidden), lin(rank + rope, hidden), lin(hidden, nh * dv),
                         torch.ones(rank, dtype=torch.bfloat16), torch.randn(nh, rank, nope).to(torch.bfloat16),
                         torch.randn(nh, dv, rank).to(torch.bfloat16), nh, nope, rope, rank, dv)
        la.fuse_projections()
        assert bool(getattr(la.o_proj.qweight, "etroit", False)) is attendu
        fusion = la.q_kv if la.q_kv is not None else la.q_a_proj
        if fusion is not None and getattr(fusion, "qweight", None) is not None:
            assert bool(getattr(fusion.qweight, "etroit", False)) is attendu
