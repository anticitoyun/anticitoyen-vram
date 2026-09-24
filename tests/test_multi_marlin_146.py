"""Pièce 146 (3) : sous la disposition Marlin unique, une MultiProjection (`_qw`/`_bs` gardent les adresses vivantes)
retenait la naturelle des poids convertis — +2,11 Gio sur Qwen3.8-27B (GDN qkv + gate, inventaire de la prise 146).
La passe la retire : après conversion, la mémoire allouée ne dépasse pas celle d'avant (une copie, jamais deux)."""
import pytest
import torch

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _lin(n, k, graine):
    from acvram.engine.layers import QuantLinear
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator(device="cuda").manual_seed(graine)
    return QuantLinear(quantize_nvfp4(torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02))


@carte
def test_multi_projection_retiree_et_naturelle_liberee(monkeypatch):
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    from acvram.kernels.gemm_dense_etroit import MultiProjection
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin") or MP.charger(compiler=False) is None:
        pytest.skip("extension ou port Marlin absents")
    monkeypatch.setattr(kernels, "_PROJ_MARLIN_DOUBLES", frozenset())
    m = torch.nn.Module()
    m.qkv, m.gate = _lin(4096, 2048, 21), _lin(2048, 2048, 22)
    m.multi = MultiProjection([m.qkv, m.gate])
    naturelle = sum(l.qweight.qweight.numel() + l.qweight.block_scale.numel() for l in (m.qkv, m.gate))
    torch.cuda.synchronize(); avant = torch.cuda.memory_allocated()
    bilan = kernels.preparer_disposition_marlin(m)
    torch.cuda.synchronize(); apres = torch.cuda.memory_allocated()
    assert bilan["seuls"] == 2 and bilan["multi_retirees"] == 1 and m.multi is None
    assert apres - avant < 0.1 * naturelle, f"naturelle retenue : +{(apres - avant) / 2**20:.1f} Mio (naturelle {naturelle / 2**20:.1f})"
