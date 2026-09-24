"""Pièce 152 (1) : en vérification spéculative d'un hybride (ngram k = 4 au défaut, q_len = 5), les projections GDN sont
faites en UNE lecture des poids (GEMV NVFP4 à q_len lignes) au lieu de q_len. Condition du chef : sortie AU BIT de
l'ancien déroulé (q_len appels à M = 1). (1) le GEMV rend chaque ligne au bit à NV = q_len contre NV = 1, aux formes de
Qwen3.8-27B ; (2) `decode_static_lignes` rend au bit sorties, états et photographies de `decode_static` jeton par jeton.
Bras cassant (prise) : le GEMM étroit rendu à ces appels (`lignes = False`) → rouge."""
import copy

import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _lin(n, k, graine):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator(device="cuda").manual_seed(graine)
    return QuantLinear(quantize_nvfp4(torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02))


def _pret():
    from acvram import kernels
    if kernels.get_extension() is None:
        pytest.skip("extension CUDA absente")
    return kernels


@carte
@pytest.mark.parametrize("n,k", [(10240, 5120), (6144, 5120), (48, 5120), (5120, 6144)])   # qkv, gate, α/β, out
@pytest.mark.parametrize("q_len", [2, 5, 8])
def test_gemv_ligne_au_bit(n, k, q_len):
    kernels = _pret()
    lin = _lin(n, k, 7)
    x = torch.randn(q_len, k, device="cuda", dtype=torch.bfloat16)
    un = torch.cat([lin(x[j:j + 1]) for j in range(q_len)])
    with kernels.gemv_par_lignes():
        lot = lin(x)
    assert torch.equal(lot, un), f"{int((lot != un).sum())} éléments diffèrent à q_len = {q_len}"


@carte
def test_decode_static_lignes_au_bit():
    _pret()
    from acvram.engine.gdn import GatedDeltaNet, _voie_fla
    H, nk, nv, dk, dv = 1024, 2, 4, 64, 64
    kd, vd = nk * dk, nv * dv
    conv_dim = 2 * kd + vd
    g = torch.Generator(device="cuda").manual_seed(3)
    la = GatedDeltaNet(_lin(conv_dim, H, 1), _lin(vd, H, 2), _lin(nv, H, 3), _lin(nv, H, 4), _lin(H, vd, 5),
                       torch.randn(conv_dim, 4, device="cuda", generator=g, dtype=torch.bfloat16) * 0.3,
                       torch.randn(nv, device="cuda", generator=g), torch.randn(nv, device="cuda", generator=g) * 0.1,
                       torch.ones(dv, device="cuda", dtype=torch.bfloat16), nk, nv, dk, dv)
    if not _voie_fla(torch.empty(1, device="cuda")):
        pytest.skip("fla absent")
    assert la.lignes_au_bit()
    q_len = 5
    h = torch.randn(q_len, H, device="cuda", dtype=torch.bfloat16)
    st = la.new_static(torch.device("cuda"))
    st["conv"].normal_(); st["S"].normal_(std=0.1)
    st_ref = {k: v.clone() for k, v in st.items()}
    hist = {k: torch.zeros((q_len,) + tuple(v.shape), device="cuda", dtype=v.dtype) for k, v in (("conv", st["conv"]), ("S", st["S"]))}
    hist_ref = copy.deepcopy(hist)
    ys = []
    for j in range(q_len):                                           # l'ancien déroulé (couches._la_decode)
        ys.append(la.decode_static(h[j:j + 1], st_ref))
        for k_, v_ in hist_ref.items():
            v_[j].copy_(st_ref[k_])
    ref = torch.cat(ys)
    y = la.decode_static_lignes(h, st, hist)
    assert torch.equal(y, ref), f"sortie : {int((y != ref).sum())} éléments diffèrent"
    for k_ in ("conv", "S"):
        assert torch.equal(st[k_], st_ref[k_]), f"état {k_}"
        assert torch.equal(hist[k_], hist_ref[k_]), f"photographie {k_}"
