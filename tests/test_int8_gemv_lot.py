"""INT8 GEMV à N activations : une passe sur les poids pour N ≤ 12.

Le profil du 14/09 (décodage b=12) montrait int8_gemv<4,8> + <4,4> à chaque
couche : NV plafonnait à 8 et les poids étaient lus deux fois. La version
élargie doit rendre, ligne par ligne, exactement ce que rend N=1 (même ordre
d'accumulation), pour N = 1..16 et K de 2048 à 5120.
"""
import pytest
import torch

from acvram.quant.formats import quantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "int8_gemv"):
        pytest.skip("extension sans int8_gemv")
    return ext


@pytest.mark.parametrize("M,K", [(5120, 2048), (2048, 4096), (4096, 2048)])
@pytest.mark.parametrize("N", [1, 2, 8, 9, 12, 13, 16])
def test_int8_gemv_lot_bit_identique_a_n1(M, K, N):
    ext = _ext()
    torch.manual_seed(M + K + N)
    w = (torch.randn(M, K, device="cuda") * 0.05)
    t = quantize(w, "int8", group_size=128)
    x = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    y = ext.int8_gemv(t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(), x.contiguous(), t.group_size)
    assert y.shape == (N, t.qweight.shape[0])
    un_par_un = torch.cat([ext.int8_gemv(t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
                                         x[i:i + 1].contiguous(), t.group_size) for i in range(N)])
    assert torch.equal(y, un_par_un), f"N={N} : {int((y != un_par_un).sum())} valeurs differentes de N=1"
    # et contre la référence déquantifiée en fp32 : tolérance bf16
    from acvram.kernels import int8_dequant
    ref = x.float() @ int8_dequant(t, torch.float16).float().T
    ecart = (y.float() - ref).abs()
    assert (ecart <= ref.abs() * 2 ** -6 + 1e-2).float().mean().item() > 0.999
