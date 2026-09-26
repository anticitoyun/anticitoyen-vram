"""Poste F, fusion (3b) : la norme d'entrée dans le GEMV int8 q/k/v
(`int8_gemv_norme`, prologue de `int8_gemv_kernel`, acvram_kernels.cu) —
RÉFUTÉ comme défaut (poste3 6dbb1bb : exact, mais +0,31 ms/pas, la norme est
recalculée par chaque bloc) ; le noyau reste un témoin nommé, ce test garde
son exactitude.

Sur carte : (y, x_out) = int8_gemv_norme(delta, W, res, w, eps, mult) contre
add_norm(res, delta) puis int8_gemv — x_out AU BIT (même formule bf16(res +
mult·delta)), y à ± 2⁻⁸ relatif à Σ|h·w| (rs = rsqrtf d'une somme fp32 dont
seul l'ordre diffère : au plus 1 ulp bf16 sur quelques h), pour N = 1, 2, 8
et K = 2048 (Coder), et un bras qui doit casser (eps × 10⁶ change y).
À sec : la règle d'éligibilité (INT8 empilé, sans AWQ ni biais, N ≤ 8,
K multiple de 16, ≤ 32 Kio) rend None et non un résultat faux."""
import pytest
import torch

from acvram import kernels
from acvram.quant.formats import INT8Tensor, _dequantize_int8, _quantize_int8

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")


def _tenseur(n, k, dev):
    g = torch.Generator().manual_seed(n + k)
    t = _quantize_int8(torch.randn(n, k, generator=g) * 0.05, group_size=128)
    t.qweight, t.scales, t.zeros = t.qweight.to(dev), t.scales.to(dev), t.zeros.to(dev)
    return t


def test_a_sec_l_ineligible_rend_none_jamais_un_faux_resultat():
    t = _tenseur(256, 2048, "cpu")
    x = torch.randn(1, 2048).to(torch.bfloat16)
    w = torch.ones(2048, dtype=torch.bfloat16)
    assert kernels.int8_matmul_norme(x, t, x, w, 1e-6) is None            # pas d'extension / cpu


@CUDA
@pytest.mark.parametrize("n", [1, 2, 8])
def test_sur_carte_y_et_x_out_suivent_add_norm_puis_gemv(n):
    from acvram.engine.layers import RMSNorm, add_norm
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "int8_gemv_norme"):
        pytest.skip("int8_gemv_norme absent de l'extension")
    K, M = 2048, 4096 + 1024
    t = _tenseur(M, K, "cuda")
    torch.manual_seed(n)
    res = torch.randn(n, K, device="cuda").to(torch.bfloat16)
    delta = (torch.randn(n, K, device="cuda") * 0.3).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(K, device="cuda")).to(torch.bfloat16)
    norme = RMSNorm(w.clone(), eps=1e-6)
    x_ref, h_ref = add_norm(res, delta, norme, 1.0)
    y_ref = kernels.int8_matmul(h_ref, t)
    r = kernels.int8_matmul_norme(delta, t, res, w, 1e-6, 1.0)
    assert r is not None
    y, x = r
    assert torch.equal(x, x_ref), "x_out doit être bf16(res + delta) au bit"
    wd = _dequantize_int8(t, torch.float64)
    borne = h_ref.double().abs() @ wd.abs().T
    ecart = (y.double() - y_ref.double()).abs()
    hors = int((ecart > 2 ** -8 * borne).sum())
    assert hors == 0, f"{hors} valeurs hors 2^-8 (max {float(ecart.max()):.3e})"
    y2, _ = kernels.int8_matmul_norme(delta, t, res, w, 1.0, 1.0)      # eps × 10⁶ : doit changer y
    assert int(((y2.double() - y_ref.double()).abs() > 2 ** -8 * borne).sum()) > 0
