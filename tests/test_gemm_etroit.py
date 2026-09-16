"""Juge du poste C (kernels/gemm_etroit.py) : le GEMM étroit W8A16 Triton
contre la référence float64 (déquant exacte de `_dequantize_int8`), à
2⁻⁸ × Σ|x·w| par valeur ; bras qui doivent casser : zéro ignoré, échelle
décalée d'un groupe, entrée plus courte que k_pad ; tête en fp32 ;
plusieurs tranches K (b = 1) et une seule. Sans carte :
``TRITON_INTERPRET=1`` en fp16.
"""
import importlib
import os

import pytest
import torch

from acvram.quant.formats import INT8Tensor, _dequantize_int8, _quantize_int8

TOL = 2 ** -8


def _ge():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    ge = importlib.import_module("acvram.kernels.gemm_etroit")
    if not ge.disponible():
        pytest.skip("Triton indisponible")
    return ge


def _montage(m, n, k, graine=0, group=128):
    torch.manual_seed(graine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(n, k) * 0.05 + torch.randn(n, 1) * 0.02
    t = _quantize_int8(w, group_size=group)
    t.qweight, t.scales, t.zeros = t.qweight.to(dev), t.scales.to(dev), t.zeros.to(dev)
    x = (torch.randn(m, k) * torch.logspace(-1, 1, m).unsqueeze(1)).to(dev)
    x = x.to(torch.bfloat16 if dev == "cuda" else torch.float16)
    return x, t


def _juger(y, x, t):
    w = _dequantize_int8(t, torch.float64)
    ref = x.double() @ w.T
    borne = x.double().abs() @ w.abs().T
    hors = ((y.double() - ref).abs() > TOL * borne).sum()
    return int(hors), ref


@pytest.mark.parametrize("m,n,k", [(12, 2048, 2048), (1, 512, 4096), (16, 5120, 1024), (3, 100, 200)])
def test_la_sortie_suit_la_reference(m, n, k):
    ge = _ge()
    x, t = _montage(m, n, k)
    y = ge.gemm_etroit(x, t)
    assert y.shape == (m, n) and y.dtype == x.dtype
    hors, _ = _juger(y, x, t)
    assert hors == 0, hors


def test_la_tete_rend_du_fp32():
    ge = _ge()
    x, t = _montage(12, 3000, 512)
    y = ge.gemm_etroit(x, t, sortie_fp32=True)
    assert y.dtype == torch.float32 and _juger(y, x, t)[0] == 0


def test_les_bras_qui_doivent_casser():
    ge = _ge()
    x, t = _montage(12, 1024, 1024)
    _, ref = _juger(ge.gemm_etroit(x, t), x, t)
    z0 = INT8Tensor(t.qweight, t.scales, torch.zeros_like(t.zeros), t.group_size, t.shape)
    assert _juger(ge.gemm_etroit(x, z0), x, t)[0] > 0, "zéro ignoré invisible"
    sd = INT8Tensor(t.qweight, t.scales.roll(1, dims=1), t.zeros, t.group_size, t.shape)
    assert _juger(ge.gemm_etroit(x, sd), x, t)[0] > 0, "échelle décalée d'un groupe invisible"


def test_l_entree_plus_courte_que_k_pad():
    ge = _ge()
    x, t = _montage(8, 256, 200)
    assert t.qweight.shape[1] > 200
    y = ge.gemm_etroit(x, t)
    assert _juger(y, x, t)[0] == 0
