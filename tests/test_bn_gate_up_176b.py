"""Pièce 176b : l'étroit int8 prend des tuiles N de 32 (au lieu de 64) pour une forme à UNE tranche K qui déborde d'à peine
une vague (gate‖up 34 816 × 5 120 : 544 tuiles pour 510 places, 1,07 vague) — même noyau, sortie AU BIT (tl.dot rend la
même somme par colonne). Le test de choix casse si la règle disparaît (le test au bit seul ne le pourrait pas : il est vert
dans les deux cas, par construction)."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


@carte
def test_regle_bn():
    from acvram.kernels import gemm_etroit as GE
    d = torch.device("cuda")
    if GE._programmes(d) != 170:
        pytest.skip("règle calibrée pour 170 SM (5090)")
    assert GE.bn_pour(34816, 40, d) == 32            # gate‖up : 1 tranche, 1,07 vague
    for n in (10240, 6144, 5120, 14336, 248320):      # qkv, gate, o/out, qkv attention, tête : 64
        assert GE.bn_pour(n, 40, d) == 64, n


@carte
@pytest.mark.parametrize("M", [2, 8, 16])
@pytest.mark.parametrize("compact", [True, False])
def test_bn32_au_bit_du_bn64(M, compact, monkeypatch):
    from acvram.kernels import gemm_etroit as GE
    from acvram.kernels import vue_g128
    from acvram.quant.formats import quantize
    if not GE.disponible():
        pytest.skip("Triton absent")
    t = vue_g128(quantize(torch.randn(34816, 5120, device="cuda", dtype=torch.bfloat16) * 0.02, "int8",
                          group_size=5120, symmetric=True))
    x = torch.randn(M, 5120, device="cuda", dtype=torch.bfloat16)
    y32 = GE.gemm_etroit(x, t, compact=compact)
    monkeypatch.setattr(GE, "bn_pour", lambda N, ng, device: GE.BN)
    y64 = GE.gemm_etroit(x, t, compact=compact)
    assert torch.equal(y32, y64), f"{int((y32 != y64).sum())} valeurs diffèrent"
