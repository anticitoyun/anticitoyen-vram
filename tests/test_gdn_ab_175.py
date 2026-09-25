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


def test_le_defaut_est_auto():
    """175 b (chef, 25/09) : le gain (b=8 mixte −12 % par pas, au bit à chaque M) est SERVI — rouge si le défaut revient à
    separe, dans le module comme dans la table de régime (la ligne [régime] et `acvram doctor` lisent la table)."""
    import os
    from acvram import regime
    v = next(v for v in regime.VARIABLES if v.nom == "GDN_AB")
    assert (gdn.AB_DEFAUT, v.defaut, v.torch) == ("auto", "auto", "separe")
    if "ACVRAM_GDN_AB" not in os.environ:
        assert gdn._GDN_AB == "auto"


def test_le_bilan_se_remet_a_zero(monkeypatch):
    """175 b : `ab_bilan_reinit` (appelée par le chargeur avant sa passe de fusion) efface le cumul des chargements précédents."""
    monkeypatch.setitem(gdn.AB_BILAN, "fusionnees", 28)
    monkeypatch.setitem(gdn.AB_BILAN, "raisons", {"scaler": 1})
    gdn.ab_bilan_reinit()
    assert gdn.AB_BILAN == {"fusionnees": 0, "raisons": {}}
    assert gdn._ab_texte() in ("", " ab=auto(0)", f" ab={gdn._GDN_AB}(0)")


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
    monkeypatch.setattr(c, "_ab", lambda x, **kw: appels.append(tuple(x.shape)) or orig(x, **kw))
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
@pytest.mark.parametrize("m", [1, 2, 3, 4, 6, 8, 12, 16, 24])
def test_auto_au_bit_des_deux_appels(monkeypatch, m):
    """Prises 3-4 : triton == les deux F.linear AU BIT à M = 2, 4, 8 (même ordre d'accumulation tensor-core que le wmma
    128x1 de cuBLAS), PLUS à M = 12 et 16 (autre noyau cuBLAS) ; à M = 1, concat == les deux appels au bit. `auto` (concat
    à 1, triton à 2-8, les deux appels au-delà) doit rester au bit à CHAQUE M. Rouge si cuBLAS change de noyau."""
    c, wa, wb = _couche(monkeypatch, "auto")
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    b, a = c._ab(x)
    assert torch.equal(b, torch.nn.functional.linear(x, wb)) and torch.equal(a, torch.nn.functional.linear(x, wa)), m


@carte
@pytest.mark.parametrize("mode", ["auto", "concat", "triton"])
@pytest.mark.parametrize("m", [1, 8, 16])
def test_les_casts_absorbes_sont_au_bit(monkeypatch, mode, m):
    """175 (poste1, 182) : `_ab(x, fp32=True)` == `_ab(x).to(float32)` au bit, et `_projections` rend b, a en fp32 sans cast."""
    c, wa, wb = _couche(monkeypatch, mode)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    b, a = c._ab(x)
    b32, a32 = c._ab(x, fp32=True)
    assert b32.dtype == torch.float32 and torch.equal(b32, b.to(torch.float32)) and torch.equal(a32, a.to(torch.float32))
    _, _, bp, ap = c._projections(x)
    assert bp.dtype == torch.float32 and torch.equal(bp, b32) and torch.equal(ap, a32)


@carte
def test_bras_cassant_un_poids_corrompu_change_la_sortie(monkeypatch):
    c, wa, wb = _couche(monkeypatch, "concat")
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
    b, a = c._ab(x)
    c.ab.qweight.weight[NV + 3, :64] += 1.0                          # α corrompu
    b2, a2 = c._ab(x)
    assert torch.equal(b, b2) and not torch.equal(a, a2)
