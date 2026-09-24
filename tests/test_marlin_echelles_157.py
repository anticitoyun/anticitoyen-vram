"""Pièce 157 : la conversion des échelles NVFP4 → Marlin (S0E5M3, ×facteur×2⁷, < 2 → 0) écrasait À ZÉRO les échelles de
bloc sous-normales d'un poids qui mêle 448 et 2⁻⁹ (Qwen3-14B couche 2 down : 47 368 valeurs fausses ; Coder-30B couche 0 :
0,57 % des blocs de 43 experts). Correctif : facteur PAR LIGNE (échelle globale par colonne) quand un facteur scalaire
écraserait ; sinon, poids exclu de la disposition (chemin naturel, compté `inexacts`). Les échelles sous-normales sont
FORCÉES ici : les poids randn × 0,02 des autres tests n'en ont pas, ce qui a caché le défaut."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
E4 = torch.float8_e4m3fn


def test_comptage_des_ecrasements_a_sec():
    from acvram.kernels import marlin_port as MP
    bs = torch.tensor([[448.0, 2 ** -9, 1.0, 0.0],         # ligne large : 448 et une sous-normale
                       [2 ** -8, 2 ** -9, 2 ** -7, 0.0]]).to(E4)   # ligne petite : tient seule
    assert MP.echelles_ecrasees(bs) == 4                  # facteur scalaire (max 448) : 2⁻⁹ ; 2⁻⁸, 2⁻⁹, 2⁻⁷ (< 2⁻⁶)
    assert MP.echelles_ecrasees(bs, par_ligne=True) == 1  # par ligne : seule la sous-normale de la ligne large reste
    assert MP.echelles_ecrasees(bs[1:], par_ligne=True) == 0


def _poids_lignes_minuscules(n, k, graine):
    """Lignes 0..n/2 d'amplitude 1, les autres × 1e-5 : leurs échelles de bloc tombent en sous-normales e4m3 (> 0)."""
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator(device="cuda").manual_seed(graine)
    w = torch.randn(n, k, device="cuda", generator=g)
    w[n // 2:] *= 1e-5
    return quantize_nvfp4(w.to(torch.bfloat16))


@carte
def test_facteur_par_ligne_depaquete_au_bit():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("port Marlin absent")
    t = _poids_lignes_minuscules(2048, 1024, 157)
    assert MP.echelles_ecrasees(t.block_scale) > 0, "montage : le facteur scalaire doit écraser des sous-normales"
    assert MP.marlin_exact(t) is None                      # par ligne : exact
    w, s_, g = MP.preparer_dense(t)
    W = MP.depaqueter_marlin(w, s_, g, t.padded_in, t.qweight.shape[0]).to(torch.bfloat16)
    ref = kernels.nvfp4_dequant(t, torch.bfloat16)
    assert torch.equal(W[:, : ref.shape[1]], ref), f"{int((W[:, : ref.shape[1]] != ref).sum())} valeurs fausses"


@carte
def test_poids_inexact_exclu_de_la_disposition(monkeypatch):
    from acvram import kernels
    from acvram.engine.layers import QuantLinear
    from acvram.kernels import marlin_port as MP
    from acvram.quant.nvfp4 import quantize_nvfp4
    if kernels.get_extension() is None or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    w = torch.randn(2048, 1024, device="cuda")
    w[:, :16] *= 1e-5                                       # un bloc SOUS-NORMAL (≈ 0,004) dans CHAQUE ligne : ligne large
    t = quantize_nvfp4(w.to(torch.bfloat16))
    assert MP.marlin_exact(t) is not None
    m = torch.nn.Module(); m.proj = QuantLinear(t)
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset())
    bilan = kernels.preparer_disposition_marlin(m)
    assert bilan.get("inexacts") == 1 and bilan["seuls"] == 0 and m.proj.qweight.qweight is not None
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
    assert kernels._marlin_dense(x, m.proj.qweight) is None, "le Marlin paresseux doit aussi laisser l'inexact au naturel"


@carte
def test_prefill_du_poids_a_lignes_minuscules_au_bit_du_naturel(monkeypatch):
    """Chemin servi : disposition unique, préfill (M > 32) = dépaquetage + F.linear — au bit du naturel."""
    from acvram import kernels
    from acvram.engine.layers import QuantLinear
    from acvram.kernels import marlin_port as MP
    if kernels.get_extension() is None or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    t = _poids_lignes_minuscules(2048, 1024, 158)
    lin = QuantLinear(t)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    ref = lin(x)
    m = torch.nn.Module(); m.proj = lin
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset())
    bilan = kernels.preparer_disposition_marlin(m)
    assert bilan["seuls"] == 1 and not bilan.get("inexacts")
    y = lin(x)
    assert torch.equal(y, ref), f"préfill Marlin ≠ naturel : max|Δ| {float((y - ref).abs().max())}"
