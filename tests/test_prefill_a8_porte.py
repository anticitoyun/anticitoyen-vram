"""Porte W4A8 (sage-w4a4-clos-w4a8-porte-19-09) : `fausse_quant_a8` et son
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


def test_porte_w8r_poids_par_ligne():
    """Porte W8r : arrondi int8 symétrique par ligne de sortie — ≤ 255 valeurs
    distinctes par ligne, échelle amax/127 par ligne (l'amax est conservé au
    bit), erreur relative sous celle de l'A4 d'activation ; variable dans la table."""
    from acvram.quant.fakequant_activation import fake_quantize_w8_row
    from acvram import regime
    torch.manual_seed(3)
    w = (torch.randn(4, 96, 256) * 0.02).to(torch.bfloat16)
    w[0, 5, 7] = 1.0                                               # un outlier par ligne : sa ligne seule paie
    y = fake_quantize_w8_row(w.clone())                            # en place : on garde w intact
    assert y.shape == w.shape and y.dtype == w.dtype
    assert torch.equal(y.float().abs().amax(-1), w.float().abs().amax(-1))          # amax de chaque ligne conservé
    assert int(torch.unique(y[1, 3].float()).numel()) <= 255
    err = (y.float() - w.float()).norm() / w.float().norm()
    assert 0.0 < err < 0.02
    assert (y[0, 5].float() - w[0, 5].float()).abs().max() > (y[0, 6].float() - w[0, 6].float()).abs().max()
    assert "ACVRAM_PREFILL_W8R" in {v.env for v in regime.VARIABLES}


def test_porte_w8r_en_place_par_blocs_egale_un_seul_tenant_et_refuse_inerte(monkeypatch):
    from acvram.quant.fakequant_activation import fake_quantize_w8_row
    torch.manual_seed(5)
    w = (torch.randn(3, 300, 128) * 0.02).to(torch.bfloat16)
    ref = fake_quantize_w8_row(w.clone(), bloc_lignes=10 ** 9)            # un seul tenant
    w2 = w.clone()
    out = fake_quantize_w8_row(w2, bloc_lignes=7)                         # par blocs, en place
    assert out is w2 and torch.equal(out, ref)                            # même résultat, aucun nouveau tenseur
    # refus nommé quand la porte serait inerte sur le chemin pris
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_marlin_refus_distinct import _bloc_distinct
    from acvram.engine import model as MD
    from acvram.engine import moe as MOE_D
    bloc = _bloc_distinct(top_k=2)
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "naturel")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "grouped_mm")
    assert bloc._try_build_stacks(), bloc._raison_repli
    monkeypatch.setattr(MOE_D, "_PREFILL_W8R", "1")
    monkeypatch.setattr(MOE_D, "_MOE_MMA", True)                             # la MMA primerait : inerte
    x = (torch.randn(4, 256) * 0.5).to(torch.bfloat16)
    logits = bloc.router(x)
    topw, topi = torch.topk(torch.softmax(logits.float(), -1), bloc.top_k, dim=-1)
    with pytest.raises(RuntimeError, match="inerte"):
        bloc._forward_prefill_grouped(x, topw, topi)
