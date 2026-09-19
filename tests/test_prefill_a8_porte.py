"""Porte W4A8 (poste7-w4a4-clos-w4a8-porte-19-09) : `fausse_quant_a8` et son
arrondi int8 `quantifier_a8_torch`, tenus AU BIT contre le noyau Triton
`_quant_a8_kernel` (sous TRITON_INTERPRET=1, dans un sous-processus : le
mode interprète se choisit avant l'import de triton), sur un tenseur à
outliers (amax/rms ≈ 20-100 comme les activations réelles de Coder,
verdict-w4a4-a-sec-19-09) et sur les demi-entiers qui séparent « au pair »
de « éloigné de zéro »."""
import os
import subprocess
import sys
import pathlib

import pytest
import torch

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from acvram.kernels.gemm_w8a8 import quantifier_a8_torch                  # noqa: E402


def _tenseur():
    g = torch.Generator().manual_seed(2026_09_19)
    x = torch.randn(64, 2048, generator=g) * 0.7
    x[::7, ::113] *= 40.0                                  # outliers : amax/rms ≈ 30
    x[3, :16] = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, -3.5,
                              126.5, -126.5, 127.4, -127.4, 200.0, -200.0, 0.0, 1e-9]) * (x[3].abs().amax() / 127.0)
    return x.to(torch.bfloat16)


def test_arrondi_int8_torch_egal_au_noyau_au_bit(tmp_path):
    x = _tenseur()
    chemin = tmp_path / "x.pt"
    torch.save(x, chemin)
    code = (
        "import sys, torch; sys.path.insert(0, %r)\n"
        "from acvram.kernels import gemm_w8a8 as W\n"
        "x = torch.load(%r)\n"
        "a, s = W.quantifier_a8(x)\n"
        "torch.save((a, s), %r)\n" % (str(RACINE), str(chemin), str(tmp_path / "noyau.pt")))
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        pytest.skip(f"noyau Triton indisponible sous l'interpréteur : {r.stderr[-300:]}")
    a_k, s_k = torch.load(tmp_path / "noyau.pt")
    a_t, s_t = quantifier_a8_torch(x)
    assert torch.equal(s_k.float(), s_t) and torch.equal(a_k, a_t)


def test_fausse_quant_a8_modes_et_regime():
    from acvram.engine.model import fausse_quant_a8, fausse_quant_nvfp4
    from acvram import regime
    x = _tenseur()
    y8 = fausse_quant_a8(x, "int8")
    assert y8.dtype == x.dtype and y8.shape == x.shape
    a, s = quantifier_a8_torch(x)
    assert torch.equal(y8, (a.float() * s[:, None]).to(x.dtype))     # reconstruction = a·s
    e8 = (y8.float() - x.float()).norm() / x.float().norm()
    e4m3 = (fausse_quant_a8(x, "e4m3").float() - x.float()).norm() / x.float().norm()
    e4 = (fausse_quant_nvfp4(x).float() - x.float()).norm() / x.float().norm()
    assert e8 < e4 and e4m3 < e4                                       # A8 < A4, les deux formats
    with pytest.raises(Exception):
        fausse_quant_a8(x, "fp4")
    noms = {v.env for v in regime.VARIABLES}
    assert {"ACVRAM_PREFILL_A8", "ACVRAM_PREFILL_A8_FMT"} <= noms
    assert "ACVRAM_PREFILL_A8" in regime.regime_noyaux()["variables"]
