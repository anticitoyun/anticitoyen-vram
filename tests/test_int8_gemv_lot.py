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
    if N <= 12:
        # même noyau (warp) pour N et pour 1 : ligne par ligne, bit-identique.
        # Au-delà de 12 le lot passe par le noyau à blocs dont l'ordre de
        # réduction diffère : seule la tolérance fp32/bf16 vaut.
        un_par_un = torch.cat([ext.int8_gemv(t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
                                             x[i:i + 1].contiguous(), t.group_size) for i in range(N)])
        assert torch.equal(y, un_par_un), f"N={N} : {int((y != un_par_un).sum())} valeurs differentes de N=1"
    # et contre la référence déquantifiée en fp32 : tolérance bf16
    from acvram.kernels import int8_dequant
    ref = x.float() @ int8_dequant(t, torch.float16).float().T
    ecart = (y.float() - ref).abs()
    assert (ecart <= ref.abs() * 2 ** -6 + 1e-2).float().mean().item() > 0.999


@pytest.mark.parametrize("M,K", [(5120, 2048), (2048, 4096), (4096, 2048), (151936, 2048)])
@pytest.mark.parametrize("N", [1, 12])
def test_int8_gemv_warp_contre_blocs(M, K, N):
    """Le noyau « un warp par ligne » (bead z5q) contre le noyau à blocs :
    l'ordre de réduction diffère, l'écart est celui d'une somme fp32
    réordonnée, borné à un ulp bf16 ; et N=12 rend ligne par ligne ce que
    rend N=1 (même noyau)."""
    import os
    ext = _ext()
    torch.manual_seed(M + K + N + 1)
    w = (torch.randn(M, K, device="cuda") * 0.05)
    t = quantize(w, "int8", group_size=128)
    x = torch.randn(N, K, device="cuda").to(torch.bfloat16)
    args = (t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous())
    y_warp = ext.int8_gemv(*args, x.contiguous(), t.group_size)
    from acvram.kernels import int8_dequant
    ref = x.float() @ int8_dequant(t, torch.float16).float().T
    ecart = (y_warp.float() - ref).abs()
    tol = ref.abs() * 2 ** -7 + 1e-3 * ref.abs().max()
    hors = int((ecart > tol).sum())
    assert hors == 0, f"{hors} hors tolerance, max {ecart.max().item():.3e}"
    if N > 1:
        un = torch.cat([ext.int8_gemv(*args, x[i:i + 1].contiguous(), t.group_size) for i in range(N)])
        assert torch.equal(y_warp, un)


@pytest.mark.parametrize("M,K", [(151936, 2048), (5120, 2048)])
@pytest.mark.parametrize("N", [1, 12, 16])
def test_sortie_fp32_egale_au_bit_au_chemin_x_fp32(M, K, N):
    """Tête lm_head (poste7-duel-verdict § 14 (ii)) : x bf16 + sortie_fp32 rend
    exactement les logits du chemin x.to(float32) (mêmes produits, même ordre
    de sommes, sans conversion de h) — et pas ceux du chemin bf16, qui arrondit
    la sortie (l'argmax basculait, § 13)."""
    ext = _ext()
    torch.manual_seed(M + K + N + 7)
    w = torch.randn(M, K, device="cuda") * 0.05
    t = quantize(w, "int8", group_size=128)
    args = (t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous())
    x = (torch.randn(N, K, device="cuda") * 3).to(torch.bfloat16).contiguous()
    y32 = ext.int8_gemv(*args, x, t.group_size, True)
    ref = ext.int8_gemv(*args, x.float().contiguous(), t.group_size)
    assert y32.dtype == torch.float32 and ref.dtype == torch.float32
    assert torch.equal(y32, ref), f"{int((y32 != ref).sum())} logits différents du chemin x fp32"
    y16 = ext.int8_gemv(*args, x, t.group_size)
    assert y16.dtype == torch.bfloat16 and not torch.equal(y16.float(), ref)
    # et par int8_matmul (le chemin de MoEModel._tete)
    from acvram.kernels import int8_matmul
    z = int8_matmul(x, t, sortie_fp32=True)
    assert z.dtype == torch.float32 and torch.equal(z, ref[:, :t.shape[0]])
