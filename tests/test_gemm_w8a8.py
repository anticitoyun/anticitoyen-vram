"""GEMM W8A8 du préfill (`kernels/gemm_w8a8.py`, P0) : sortie = s_x · (a8 ⊗
déquant(w)) en fp32 à 2⁻⁷ × Σ|a·w| près (le produit entier est exact, seule
l'accumulation des groupes en fp32 diffère) ; l'A8 par jeton est l'unique
source d'écart contre x bf16 (mesurée, pas jugée) ; bras cassant : zéro
ignoré → rouge. Sans carte : interpréteur Triton."""
import pytest
import torch

from acvram.kernels import gemm_w8a8 as W
from acvram.quant.formats import _dequantize_int8, _quantize_int8

pytestmark = pytest.mark.skipif(not W.disponible(), reason="Triton absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if torch.cuda.is_available() else torch.float16
TOL = 2 ** -7


def _cas(M, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    t = _quantize_int8((torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16), 128)
    t.qweight, t.scales, t.zeros = t.qweight.to(DEV), t.scales.to(DEV), t.zeros.to(DEV)
    x = (torch.randn(M, K, generator=g) * (1 + torch.rand(M, 1, generator=g) * 4)).to(DT).to(DEV)
    return x, t


def _reference(x, t):
    a, s = W.quantifier_a8(x)
    xa = a.float() * s[:, None]                                  # l'activation quantifiée, en fp32
    wd = _dequantize_int8(t, torch.float32)[:, : x.shape[1]].to(DEV)
    return xa @ wd.T, xa.abs() @ wd.abs().T


@pytest.mark.parametrize("M,K,N", [(256, 2048, 512), (300, 512, 384), (129, 1024, 256), (16, 256, 128)])   # petites : interpréteur
def test_w8a8_egale_le_produit_fp32_de_l_activation_quantifiee(M, K, N):
    x, t = _cas(M, K, N, seed=M + N)
    y = W.gemm_w8a8(x, t)
    attendu, borne = _reference(x, t)
    assert y.shape == (M, N) and y.dtype == x.dtype
    assert int(((y.float() - attendu).abs() > TOL * borne).sum()) == 0


def test_l_a8_par_jeton_est_la_seule_erreur_et_elle_est_bornee():
    x, t = _cas(128, 1024, 256, seed=3)
    y = W.gemm_w8a8(x, t).float()
    wd = _dequantize_int8(t, torch.float32)[:, :1024].to(DEV)
    exact = x.float() @ wd.T
    rel = ((y - exact).norm() / exact.norm()).item()
    assert rel < 0.01, rel                                       # ~0,4 % attendu (int8 symétrique par jeton)


def test_bras_cassant_zero_ignore():
    x, t = _cas(128, 512, 256, seed=9)
    attendu, borne = _reference(x, t)
    t.zeros = torch.full_like(t.zeros, 128)                      # « pas de zéro » : faux pour un affine
    y = W.gemm_w8a8(x, t)
    assert int(((y.float() - attendu).abs() > TOL * borne).sum()) > 0


def test_int8_matmul_emprunte_a8_sous_le_regime(monkeypatch):
    from acvram import kernels
    x, t = _cas(300, 512, 256, seed=5)                           # n = 300 > INT8_GEMV_MAX : GEMM
    monkeypatch.setattr(kernels, "_PREFILL_INT8", "a8")
    y = kernels.int8_matmul(x, t)
    assert torch.equal(y, W.gemm_w8a8(x, t)[:, :256])
    monkeypatch.setattr(kernels, "_PREFILL_INT8", "bf16")
    y0 = kernels.int8_matmul(x, t)                               # la déquant entière + linear
    assert not torch.equal(y, y0) and ((y.float() - y0.float()).norm() / y0.float().norm()).item() < 0.01
    assert kernels.prefill_int8_regime() == "bf16"
