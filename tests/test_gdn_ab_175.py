"""Pièce 175 (poste6, 25/09) : portes α et β bf16 des couches GDN en un appel (ACVRAM_GDN_AB=concat | triton).
concat doit rester AU BIT des deux F.linear (test rouge sinon, et rouge si le chemin fusionné n'est plus pris) ;
triton : ≤ 1 ulp bf16 de la référence fp32, déterministe."""
import types

import pytest
import torch

from acvram.engine import gdn
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import PlainTensor

NV, K = 48, 5120
carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _lin(w):
    return QuantLinear(PlainTensor(w, tuple(w.shape), "bf16"), None, None, w.shape[0], w.shape[1])


def _couche(monkeypatch, mode, device="cuda"):
    monkeypatch.setattr(gdn, "_GDN_AB", mode)
    torch.manual_seed(175)
    wa = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    wb = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    c = types.SimpleNamespace(alpha=_lin(wa), beta_proj=_lin(wb), nv=NV)
    c.ab = gdn.GatedDeltaNet._fusionner_ab(c)
    return c, wa, wb


def test_a_sec_ou_separe_rien_ne_change(monkeypatch):
    c, _, _ = _couche(monkeypatch, "concat", device="cpu")       # pas sur la carte : pas de fusion
    assert c.ab is None
    monkeypatch.setattr(gdn, "_GDN_AB", "separe")
    assert gdn.GatedDeltaNet._fusionner_ab(c) is None
    monkeypatch.setattr(gdn, "_GDN_AB", "inconnu")
    with pytest.raises(ValueError):
        gdn.GatedDeltaNet._fusionner_ab(c)


@carte
@pytest.mark.parametrize("m", [1, 8, 16])
def test_concat_au_bit_des_deux_appels(monkeypatch, m):
    c, wa, wb = _couche(monkeypatch, "concat")
    assert c.ab is not None and tuple(c.ab.qweight.weight.shape) == (2 * NV, K)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    b, a = gdn.GatedDeltaNet._ab(c, x)
    b_ref, a_ref = torch.nn.functional.linear(x, wb), torch.nn.functional.linear(x, wa)
    assert torch.equal(b, b_ref) and torch.equal(a, a_ref), "concat n'est plus au bit des deux F.linear"


@carte
def test_le_chemin_fusionne_est_pris(monkeypatch):
    c, _, _ = _couche(monkeypatch, "concat")
    appels = []
    orig = c.ab.forward
    monkeypatch.setattr(c.ab, "forward", lambda x: appels.append(x.shape) or orig(x))
    gdn.GatedDeltaNet._ab(c, torch.randn(8, K, device="cuda", dtype=torch.bfloat16))
    assert appels == [(8, K)], "le chemin fusionné n'est pas celui pris"


@carte
@pytest.mark.parametrize("m", [1, 8, 16])
def test_triton_a_un_ulp_et_deterministe(monkeypatch, m):
    from acvram.kernels.gemv_bf16_etroit import gemv_bf16_etroit
    c, wa, wb = _couche(monkeypatch, "triton")
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    w = c.ab.qweight.weight
    y = gemv_bf16_etroit(x, w)
    ref = x.float() @ w.float().t()
    ulp = (ref.abs().clamp_min(1e-6) / 128)                          # 1 ulp bf16 relatif (8 bits de mantisse)
    assert ((y.float() - ref).abs() <= ulp + 1e-6).all(), float(((y.float() - ref).abs() / ulp).max())
    assert torch.equal(y, gemv_bf16_etroit(x, w))
    b, a = gdn.GatedDeltaNet._ab(c, x)
    assert torch.equal(torch.cat([b, a], 1), y)
    y_cublas = torch.nn.functional.linear(x, w)
    assert ((y.float() - y_cublas.float()).abs() <= 2 * ulp + 1e-6).all()


@carte
def test_bras_cassant_un_poids_corrompu_change_la_sortie(monkeypatch):
    c, wa, wb = _couche(monkeypatch, "concat")
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    b, a = gdn.GatedDeltaNet._ab(c, x)
    c.ab.qweight.weight[NV + 3, :64] += 1.0                          # α corrompu
    b2, a2 = gdn.GatedDeltaNet._ab(c, x)
    assert torch.equal(b, b2) and not torch.equal(a, a2)
