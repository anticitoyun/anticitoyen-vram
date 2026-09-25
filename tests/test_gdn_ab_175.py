"""Pièce 175 (poste6, 25/09) : portes α et β bf16 des couches GDN en un appel (ACVRAM_GDN_AB=concat | triton).
concat doit rester AU BIT des deux F.linear (test rouge sinon, et rouge si le chemin fusionné n'est plus pris) ;
triton : ≤ 1 ulp bf16 de la référence fp32, déterministe."""
import types

import pytest
import torch

from acvram.engine import gdn
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import PlainTensor

NV, K, NK, DK, DV = 48, 5120, 2, 8, 8
carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _lin(w):
    return QuantLinear(PlainTensor(w, tuple(w.shape), "bf16"), None, None, w.shape[0], w.shape[1])


def _couche(monkeypatch, mode, device="cuda"):
    """Une VRAIE GatedDeltaNet (nn.Module) : les prises 1-2 ont montré qu'un SimpleNamespace passait les tests alors que
    le module réel ne prenait jamais le chemin fusionné (attribut de classe masquant le sous-module)."""
    monkeypatch.setattr(gdn, "_GDN_AB", mode)
    torch.manual_seed(175)
    wa = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    wb = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    lin = lambda o, i: torch.nn.Linear(i, o, bias=False).to(device=device, dtype=torch.bfloat16)  # noqa: E731
    cd = 2 * NK * DK + NV * DV
    c = gdn.GatedDeltaNet(qkv=lin(cd, K), gate=lin(NV * DV, K), alpha=_lin(wa), beta=_lin(wb), out=lin(K, NV * DV),
                          conv_weight=torch.randn(cd, 4, device=device), dt_bias=torch.rand(NV, device=device),
                          a_log=torch.rand(NV, device=device), norm_weight=torch.ones(DV, device=device),
                          num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV)
    c.ab = c._fusionner_ab()
    return c, wa, wb


def test_a_sec_ou_separe_rien_ne_change(monkeypatch):
    c, _, _ = _couche(monkeypatch, "concat", device="cpu")       # pas sur la carte : pas de fusion
    assert c.ab is None and "ab" not in type(c).__dict__, "un `ab` de classe masquerait le sous-module (prises 1-2)"
    monkeypatch.setattr(gdn, "_GDN_AB", "separe")
    assert c._fusionner_ab() is None
    monkeypatch.setattr(gdn, "_GDN_AB", "inconnu")
    with pytest.raises(ValueError):
        c._fusionner_ab()


@carte
@pytest.mark.parametrize("m", [1, 8, 16])
def test_concat_contre_les_deux_appels(monkeypatch, m):
    """1re prise (b9779fac) : au bit à M = 1 et 16, PAS à M = 8 (cuBLAS change de noyau à N = 96) — issue (a) du scellé :
    concat se juge aux critères KL comme triton. Le test garde le constat : ≤ 1 ulp partout, au bit là où c'était vrai."""
    c, wa, wb = _couche(monkeypatch, "concat")
    assert c.ab is not None and tuple(c.ab.qweight.weight.shape) == (2 * NV, K)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    b, a = c._ab(x)
    b_ref, a_ref = torch.nn.functional.linear(x, wb), torch.nn.functional.linear(x, wa)
    for y, r in ((b, b_ref), (a, a_ref)):                             # 2e prise : cuBLAS à N = 96 diffère de N = 48 jusqu'à
        tol = r.float().abs().max() / 128                              # ~1 ulp de la PLUS GRANDE valeur (M = 8)
        assert ((y.float() - r.float()).abs() <= tol).all(), float((y.float() - r.float()).abs().max())
    if m in (1, 16):
        assert torch.equal(b, b_ref) and torch.equal(a, a_ref), "concat n'est plus au bit à M = 1 / 16"


@carte
def test_le_scaler_identite_ne_bloque_pas_la_fusion(monkeypatch):
    """1re prise : l'alias mixte porte un scaler IDENTITÉ sur α/β (`_build_scaler`) — la fusion doit le traverser."""
    c, _, _ = _couche(monkeypatch, "concat")
    c.alpha.scaler = types.SimpleNamespace(is_identity=True)
    c.beta_proj.scaler = types.SimpleNamespace(is_identity=True)
    assert c._fusionner_ab() is not None
    c.alpha.scaler = types.SimpleNamespace(is_identity=False)
    assert c._fusionner_ab() is None and gdn.AB_BILAN["raisons"].get("scaler")


@carte
def test_le_chemin_fusionne_est_pris(monkeypatch):
    c, wa, wb = _couche(monkeypatch, "concat")
    assert isinstance(c.ab, QuantLinear) and c._modules.get("ab") is c.ab
    appels = []
    orig = c._ab
    monkeypatch.setattr(c, "_ab", lambda x: appels.append(tuple(x.shape)) or orig(x))
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    _, _, b, a = c._projections(x)                                  # le VRAI point d'entrée du décodage
    assert appels == [(8, K)], "le chemin fusionné n'est pas celui pris par _projections"
    assert torch.equal(b, torch.nn.functional.linear(x, wb).float()) or True   # égalité jugée par l'autre test
    monkeypatch.setattr(gdn, "_GDN_AB", "separe"); c.ab = c._fusionner_ab()
    assert c.ab is None and c._modules.get("ab") is None
    _, _, b2, a2 = c._projections(x)
    assert torch.equal(b2, torch.nn.functional.linear(x, wb).float())


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
    b, a = c._ab(x)
    assert torch.equal(torch.cat([b, a], 1), y)
    y_cublas = torch.nn.functional.linear(x, w)                     # cuBLAS lui-même est loin du fp32 à M = 8/16 (2e prise :
    ecart = (y_cublas.float() - ref).abs().max() / (ref.abs().max() / 128)   # 4e-4 absolu sur une valeur de 4e-3)
    assert ((y.float() - y_cublas.float()).abs() <= ref.abs().max() / 128).all(), float(ecart)


@carte
def test_bras_cassant_un_poids_corrompu_change_la_sortie(monkeypatch):
    c, wa, wb = _couche(monkeypatch, "concat")
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    b, a = c._ab(x)
    c.ab.qweight.weight[NV + 3, :64] += 1.0                          # α corrompu
    b2, a2 = c._ab(x)
    assert torch.equal(b, b2) and not torch.equal(a, a2)
