"""GEMM W4A16 dense à petit M (`kernels/gemm_dense_etroit.py`) contre la
déquantification de référence : le juge de B1 (2⁻⁷ × Σ|x·w|, `test_gemm_
grouped_w4a16`), formes q/k/v/o et gate/up/down de Qwen3.8 réduites, M = 2,
12, 16, 17, 32, entrée plus courte que `padded_in`, échelle globale par
ligne ; bras cassant de poste7 : l'échelle de bloc décalée d'un rang → rouge.
Sans carte : interpréteur Triton, fp16."""
import pytest
import torch

from acvram.kernels import gemm_dense_etroit as GD
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

pytestmark = pytest.mark.skipif(not GD.disponible(), reason="Triton absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if torch.cuda.is_available() else torch.float16
TOL_REL = 2 ** -7


def _cas(M, K, N, seed=0, gsr=False):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    t = quantize_nvfp4(w)
    if gsr:
        t.global_scale_rows = (t.global_scale.float() * torch.linspace(0.5, 2.0, N)).contiguous()
    x = torch.randn(M, K, generator=g).to(DT)
    t.qweight, t.block_scale, t.global_scale = t.qweight.to(DEV), t.block_scale.to(DEV), t.global_scale.to(DEV)
    if gsr:
        t.global_scale_rows = t.global_scale_rows.to(DEV)
    return x.to(DEV), t


def _juger(y, x, t):
    wd = dequantize_nvfp4(t, torch.float32)[:, : x.shape[1]]
    attendu = x.float() @ wd.T
    borne = x.float().abs() @ wd.abs().T
    return int(((y.float() - attendu).abs() > TOL_REL * borne).sum())


@pytest.mark.parametrize("M", [2, 12, 16, 17, 32])
@pytest.mark.parametrize("K,N", [(256, 320), (512, 192)])       # q/k/v/o (N = K) et gate/up (N > K) réduits
def test_la_gemm_dense_etroite_egale_la_dequant(M, K, N):
    x, t = _cas(M, K, N, seed=M + N)
    y = GD.gemm_dense_etroit(x, t, bn=64, bk=64)
    assert y.shape == (M, N) and y.dtype == x.dtype
    assert _juger(y, x, t) == 0


def test_entree_plus_courte_que_padded_in_et_echelle_par_ligne():
    x, t = _cas(12, 200, 96, seed=5, gsr=True)
    assert t.padded_in > 200
    y = GD.gemm_dense_etroit(x, t, bn=32, bk=64)
    assert _juger(y, x, t) == 0


def test_plusieurs_tranches_k_se_somment(monkeypatch):
    x, t = _cas(12, 1024, 64, seed=9)
    monkeypatch.setattr(GD, "_PROGRAMMES_PAR_SM", 64)             # force plusieurs tranches K
    tranches, par = GD._tranches(64, 1024, x.device, 64, 64)
    assert tranches > 1 and par % 64 == 0
    assert _juger(GD.gemm_dense_etroit(x, t, bn=64, bk=64), x, t) == 0


def test_bras_cassant_echelle_de_bloc_decalee_d_un_rang():
    x, t = _cas(12, 256, 128, seed=3)
    bs = t.block_scale.view(torch.uint8)
    t.block_scale = torch.roll(bs, 1, dims=1).view(torch.float8_e4m3fn)
    x2, t2 = _cas(12, 256, 128, seed=3)
    y = GD.gemm_dense_etroit(x, t, bn=64, bk=64)
    assert _juger(y, x2, t2) > 0, "l'échelle décalée d'un rang doit se voir"
