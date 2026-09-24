"""Pièce 152 (1) : en vérification spéculative d'un hybride (ngram k = 4 au défaut, q_len = 5), les projections GDN sont
regroupées PAR POIDS (q_len appels M = 1 consécutifs : le poids reste en L2) au lieu d'être déroulées jeton par jeton.
Condition du chef : sortie AU BIT de l'ancien déroulé, noyau NV = 1 inchangé (aucun .cu touché). Le test compare sorties,
états et photographies de `decode_static_lignes` à `decode_static` jeton par jeton. Bras cassant (prise) : qkv projeté
en UN appel à M = q_len (le GEMM étroit, pas au bit : 152 du 24/09) → rouge."""
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
    q_len = 5
    h = torch.randn(q_len, H, device="cuda", dtype=torch.bfloat16)
    st = la.new_static(torch.device("cuda"))
    st["conv"].normal_(); st["S"].normal_(std=0.1)
    st_ref = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
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
