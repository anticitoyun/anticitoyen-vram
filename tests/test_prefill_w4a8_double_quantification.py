"""Le prefill d'une QuantLinear NVFP4 sous extension n'était pas W4A16
jusqu'au 17/09 : au-delà de ACVRAM_NVFP4_GEMV_MAX (32) lignes,
`kernels.nvfp4_matmul` prenait par défaut `ACVRAM_PREFILL=a8` →
`fp4_gemm.nvfp4_mm_w4a8`, qui requantifie en E4M3 PAR LIGNE les activations
ET le poids déquantifié (double quantification FP4 bloc 16 → E4M3 ligne
entière), puis `torch._scaled_mm`. Depuis sage-prefill-a8-verdict-17-09 le
défaut est `bf16` (déquant exacte, cuBLAS) et ce chemin se demande par son
nom : `ACVRAM_PREFILL=w8a8`. Le régime tout-torch (ACVRAM_DISABLE_KERNELS=1,
backend « reference ») fait dequantize_nvfp4 → linear bf16 exact. C'est la seule différence porteuse
d'erreur entre les deux chemins de prefill MoE : les experts routés passent
par la pile bf16 + torch._grouped_mm dans les deux régimes
(model.py _forward_prefill_grouped), l'expert partagé et la couche dense par
celui-ci (verdict-bissection-w4a16-suite-17-09 : tout-torch 1,010 contre
noyaux 1,028).

Mesuré à sec sur les poids réels de GLM-4.7-Flash-vllm-direct
(outils/w4a8-expert-partage-17-09.py) : requantification du poids 2,5-3,1 %
RMS, activations 2,6 %, sortie W8A8 3,6-4,0 % — sur chaque projection de
l'expert partagé, à chaque couche, chaque jeton.

Témoins : (1) processeur, même arithmétique que fp4_gemm.py : la double
quantification du poids coûte > 1,5 % RMS là où la déquantification exacte
en bf16 en coûte < 0,4 % ; (2) carte : `nvfp4_matmul` à 64 lignes, bras a8
contre bras bf16 contre référence fp64 ; (3) carte : le DÉFAUT vaut le bras
bf16 (cassait tant que le défaut était a8)."""
import pytest
import torch

from acvram.kernels.fp4_gemm import _F8_MAX
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

M, K, N = 256, 1024, 64


def _e4m3_par_ligne(a: torch.Tensor) -> torch.Tensor:
    """fp4_gemm.py nvfp4_mm_w4a8 lignes sx/sw, x8/w8 : amax/448 par ligne, clamp, E4M3."""
    s = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / _F8_MAX
    return (a / s).clamp(-_F8_MAX, _F8_MAX).to(torch.float8_e4m3fn).float() * s


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _poids(dev, graine=11):
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(M, K, generator=g)
    w.view(M, K // 16, 16)[:, ::5] *= 0.05         # échelles de bloc étagées, comme un poids réel
    return quantize_nvfp4(w.to(dev))


def test_double_quantification_e4m3_par_ligne_coute_plus_que_bf16():
    t = _poids(torch.device("cpu"))
    w = dequantize_nvfp4(t, torch.float32)                # exact : code × bloc × global, fp32
    w_bf16 = dequantize_nvfp4(t, torch.bfloat16).float()  # le chemin « reference »
    w8 = _e4m3_par_ligne(w)                               # le chemin a8 (poids)
    e_bf16, e_w8 = _rel(w_bf16, w), _rel(w8, w)
    assert e_bf16 < 0.004, e_bf16                         # bf16 : 2^-9 relatif, ~0,2 %
    assert e_w8 > 0.015, e_w8                             # E4M3 par ligne : 3 bits de mantisse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="chemin _scaled_mm (carte ≥ 8.9)")
@pytest.mark.parametrize("mode,attendu", [("bf16", "exact"), ("w8a8", "lossy")])
def test_nvfp4_matmul_prefill_bras_a8_contre_bf16(monkeypatch, mode, attendu):
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None:
        pytest.skip("extension absente")
    dev = torch.device("cuda:0")
    t = _poids(dev)
    g = torch.Generator().manual_seed(12)
    x = torch.randn(N, K, generator=g).to(dev, torch.bfloat16)
    ref = x.double() @ dequantize_nvfp4(t, torch.float32).double().T
    monkeypatch.setenv("ACVRAM_PREFILL", mode)
    y = kernels.nvfp4_matmul(x, t)
    e = _rel(y, ref)
    if attendu == "exact":
        assert e < 0.005, f"bras bf16 : {e:.4f}"
    else:
        assert e > 0.015, f"bras w8a8 : {e:.4f} — le chemin W4A8 ne perd plus ?"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="chemin _scaled_mm (carte ≥ 8.9)")
def test_defaut_prefill_egale_le_bras_bf16(monkeypatch):
    from acvram import kernels
    if kernels.get_extension() is None:
        pytest.skip("extension absente")
    dev = torch.device("cuda:0")
    t = _poids(dev)
    g = torch.Generator().manual_seed(12)
    x = torch.randn(N, K, generator=g).to(dev, torch.bfloat16)
    monkeypatch.setenv("ACVRAM_PREFILL", "bf16")
    y_bf16 = kernels.nvfp4_matmul(x, t)
    monkeypatch.delenv("ACVRAM_PREFILL")
    y_def = kernels.nvfp4_matmul(x, t)
    assert _rel(y_def, y_bf16) < 0.005
