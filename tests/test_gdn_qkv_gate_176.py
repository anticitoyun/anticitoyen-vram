"""Pièce 176 : GDN qkv‖gate INT8 en UNE pile au décodage (même entrée). L'étroit Triton découpe K selon le nombre de
tuiles (decouper_k : qkv 3 tranches [14,14,12], gate 4 [10×4], une pile commune 2 [20,20]) : la pile garde la partition
de CHAQUE segment (`_segments`, `_etroit_segments_kernel`) et rend chaque colonne AU BIT des deux appels séparés, à b=1
(GEMV, sans découpage à ces formes) comme à 2 ≤ b ≤ 16. Le témoin « partition commune » doit différer (sinon le test ne
garde rien) ; le préfill (M > 16) reste en deux appels sur les vues."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _lin(n, k, graine, par_canal=True):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.formats import quantize
    g = torch.Generator(device="cuda").manual_seed(graine)
    w = torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
    return QuantLinear(quantize(w, "int8", group_size=k if par_canal else 128, symmetric=par_canal))


def _pret():
    from acvram import kernels
    from acvram.kernels import gemm_etroit
    if kernels.get_extension() is None or not gemm_etroit.disponible():
        pytest.skip("extension ou Triton absents")


@carte
@pytest.mark.parametrize("regime", ["defaut", "0"])         # 195 b : canal K entier au défaut ; 0 = témoin (tranches)
@pytest.mark.parametrize("par_canal", [True, False])
@pytest.mark.parametrize("M", [1, 2, 5, 8, 16])
def test_pile_a_segments_au_bit_des_deux_appels(M, par_canal, regime, monkeypatch):
    """Au défaut (195 b, `ACVRAM_ETROIT_CANAL=1`), la pile et ses deux segments passent tous trois par le noyau K entier
    avec la géométrie DE LA PILE (`GEOMETRIE_CANAL[(10240, 5120)]` = `[(6144, 5120)]` = `[(16384, 5120)]`) : à K entier,
    une colonne ne dépend que de sa partition K, pas de sa tuile N — au bit par construction. À 0 (témoin nommé), le
    noyau à tranches garde la partition de chaque segment (`_etroit_segments_kernel`) : au bit aussi."""
    _pret()
    if regime == "0":
        monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "0")
    else:
        monkeypatch.delenv("ACVRAM_ETROIT_CANAL", raising=False)
    from acvram.engine.layers import stack_int8_linears
    qkv, gate = _lin(10240, 5120, 1, par_canal), _lin(6144, 5120, 2, par_canal)       # formes GDN de Qwen3.8
    x = torch.randn(M, 5120, device="cuda", dtype=torch.bfloat16)
    ref = torch.cat([qkv(x), gate(x)], dim=-1)
    pile = stack_int8_linears([qkv, gate])
    pile.qweight._segments = (10240, 6144)
    y = pile(x)
    assert torch.equal(y, ref), f"M={M} : {int((y != ref).sum())} valeurs diffèrent"


@carte
def test_temoin_partition_commune_differe(monkeypatch):
    """Témoin du régime à tranches (`ACVRAM_ETROIT_CANAL=0`) : sans `_segments`, la pile prend la partition commune
    (2 tranches) et DOIT différer à b=8 — sinon l'équivalence ci-dessus ne prouverait rien. Au défaut (canal, K entier) il
    n'y a plus de partition : le témoin de ce régime est `test_temoin_canal_geometrie_differente`."""
    _pret()
    monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "0")
    from acvram.engine.layers import stack_int8_linears
    qkv, gate = _lin(10240, 5120, 1), _lin(6144, 5120, 2)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)
    ref = torch.cat([qkv(x), gate(x)], dim=-1)
    pile = stack_int8_linears([qkv, gate])
    assert not torch.equal(pile(x), ref), "la partition commune rend le même résultat : le test d'équivalence est aveugle"


@carte
def test_temoin_canal_differe_du_temoin_tranches(monkeypatch):
    """Témoin du régime au défaut (195 b) : la pile en canal (K entier) DOIT différer des deux segments calculés par le
    noyau à tranches (`ACVRAM_ETROIT_CANAL=0`) — sinon l'équivalence au défaut ne distinguerait pas les noyaux. (Constat
    du 25/09 : une autre tuile K, `64,128,8,2`, rend la pile AU BIT — `acc += tl.dot` garde une chaîne MMA continue sur K,
    l'ordre des sommes ne dépend pas de BK ; ce n'est donc pas un témoin.)"""
    _pret()
    from acvram.engine.layers import stack_int8_linears
    qkv, gate = _lin(10240, 5120, 1), _lin(6144, 5120, 2)
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "0")
    tranches = torch.cat([qkv(x), gate(x)], dim=-1)
    monkeypatch.delenv("ACVRAM_ETROIT_CANAL", raising=False)
    pile = stack_int8_linears([qkv, gate])
    pile.qweight._segments = (10240, 6144)
    assert torch.equal(pile(x), torch.cat([qkv(x), gate(x)], dim=-1))
    assert not torch.equal(pile(x), tranches), "le canal rend les bits du noyau à tranches : le témoin est aveugle"


@carte
@pytest.mark.parametrize("M", [1, 8, 64])
def test_projections_gdn_au_bit(M, monkeypatch):
    """GatedDeltaNet INT8 : `_projections` avec la pile (défaut) = sans (ACVRAM_GDN_QKV_GATE=0), au bit ; M = 64 passe
    par les vues (préfill)."""
    _pret()
    from acvram.engine.gdn import GatedDeltaNet
    H, nk, nv, dk, dv = 1024, 4, 8, 128, 128
    kd, vd = nk * dk, nv * dv

    def gdn(fusion):
        monkeypatch.setenv("ACVRAM_GDN_QKV_GATE", "1" if fusion else "0")
        g = torch.Generator(device="cuda").manual_seed(3)
        la = GatedDeltaNet(_lin(2 * kd + vd, H, 1), _lin(vd, H, 2), _lin(nv, H, 3), _lin(nv, H, 4), _lin(H, vd, 5),
                           torch.randn(2 * kd + vd, 4, device="cuda", generator=g, dtype=torch.bfloat16) * 0.3,
                           torch.randn(nv, device="cuda", generator=g), torch.randn(nv, device="cuda", generator=g) * 0.1,
                           torch.ones(dv, device="cuda", dtype=torch.bfloat16), nk, nv, dk, dv)
        la.fuse()
        return la

    avec, sans = gdn(True), gdn(False)
    assert getattr(avec, "qkv_gate", None) is not None and getattr(sans, "qkv_gate", None) is None
    x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    for a, b in zip(avec._projections(x), sans._projections(x)):
        assert torch.equal(a, b), f"M={M} : {int((a != b).sum())} valeurs diffèrent"
