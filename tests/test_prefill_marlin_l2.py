"""Pièce 147 L2 (poste6, 24/09) : ACVRAM_PREFILL=marlin — au préfill d'un poids en disposition Marlin seule, la GEMM
Marlin lit les tuiles au lieu de dépaqueter (opt-in, ± 1 ulp, jamais au défaut). Carte et port requis."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _ecart(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


@pytest.fixture
def poids(monkeypatch):
    from acvram import kernels
    from acvram.engine.layers import QuantLinear
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4
    if kernels.get_extension() is None or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    monkeypatch.setattr(kernels, "_PROJ_MARLIN", True)
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset())
    torch.manual_seed(147)
    boite = torch.nn.Module()
    qt = quantize_nvfp4(torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16) * 0.02)
    w_ref = dequantize_nvfp4(qt, torch.float32).clone()      # AVANT la disposition : la naturelle est libérée après
    boite.proj = QuantLinear(qt)
    bilan = kernels.preparer_disposition_marlin(boite)
    assert bilan["seuls"] == 1 and getattr(boite.proj.qweight, "_marlin_unique", False), bilan
    x = torch.randn(640, 4096, device="cuda", dtype=torch.bfloat16)
    return boite.proj.qweight, x, w_ref


def _passe(t, x, regime, monkeypatch):
    from acvram import kernels
    monkeypatch.setenv("ACVRAM_PREFILL", regime)
    avant = dict(kernels.CHEMINS_NVFP4)
    y = kernels.nvfp4_matmul(x, t)
    torch.cuda.synchronize()
    delta = {k: v - avant.get(k, 0) for k, v in kernels.CHEMINS_NVFP4.items() if v != avant.get(k, 0)}
    return y, delta


def test_le_chemin_est_celui_demande(poids, monkeypatch):
    from acvram import kernels
    t, x, w_ref = poids
    _, d = _passe(t, x, "bf16", monkeypatch)
    assert d == {"marlin_depaquete_prefill": 1}, d
    _, d = _passe(t, x, "marlin", monkeypatch)
    assert d == {"marlin_gemm_prefill": 1}, d
    monkeypatch.setattr(kernels, "_PREFILL_MARLIN_MAX_M", 100)
    _, d = _passe(t, x, "marlin", monkeypatch)
    assert d == {"marlin_depaquete_prefill": 1}, "au-delà de PREFILL_MARLIN_MAX_M, le dépaquetage reprend"
    monkeypatch.setenv("ACVRAM_PREFILL", "inconnu")
    with pytest.raises(ValueError):
        kernels.nvfp4_matmul(x, t)


def test_proche_du_defaut_pas_au_bit(poids, monkeypatch):
    t, x, w_ref = poids
    y_b, _ = _passe(t, x, "bf16", monkeypatch)
    y_m, _ = _passe(t, x, "marlin", monkeypatch)
    ref = x.float() @ w_ref.t()
    e_b, e_m = _ecart(y_b, ref), _ecart(y_m, ref)
    assert e_m <= 2 * e_b + 1e-3, (e_m, e_b)
    assert _ecart(y_m, y_b) < 2e-2, _ecart(y_m, y_b)
    assert not torch.equal(y_m, y_b), "le scellé dit ± 1 ulp : une égalité au bit serait un autre chemin"


def test_bras_cassant_la_gemm_lit_les_tuiles(poids, monkeypatch):
    t, x, w_ref = poids
    y_m, _ = _passe(t, x, "marlin", monkeypatch)
    w = t._marlin_dense[0]
    vue = w.view(torch.uint8)
    sauve = vue[:, :256].clone()
    vue[:, :256] ^= 0x55
    y_c, _ = _passe(t, x, "marlin", monkeypatch)
    vue[:, :256] = sauve
    assert _ecart(y_c, y_m) > 1e-2, "une tuile Marlin corrompue n'a pas changé la sortie : le chemin ne lit pas les tuiles"
    y_r, _ = _passe(t, x, "marlin", monkeypatch)
    assert torch.equal(y_r, y_m)
