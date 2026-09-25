"""Pièce 183 (opt-in `ACVRAM_INT8_GEMV_MAX_PARTAGE`) : dans la portée de B', un linéaire int8 à 16 < M ≤ 80 prend la
déquant partagée + GEMM au lieu du GEMV ; HORS portée (une séquence seule, décodage), rien ne change. Les deux
arithmétiques diffèrent (ce n'est pas au bit) : la qualité se juge par KL au scellé ; ici, le chemin pris, sa portée,
et l'écart borné à l'ordre de l'arrondi bf16. Bras cassant (prise) : condition de portée retirée → ROUGE."""
import pytest
import torch

from acvram import kernels
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import _quantize_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
DEV = torch.device("cuda:0")


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin():
    torch.manual_seed(183)
    return QuantLinear(_quantize_int8(torch.randn(6144, 5120, device=DEV) * 0.02, group_size=128))


def _x(n=78):
    g = torch.Generator(device="cpu").manual_seed(n)
    return torch.randn(n, 5120, generator=g).to(DEV, torch.bfloat16)


def _chemins(f):
    c = dict(kernels.CHEMINS_INT8)
    y = f()
    return y, {k: v - c.get(k, 0) for k, v in kernels.CHEMINS_INT8.items() if v != c.get(k, 0)}


def test_dans_la_portee_deq_partagee_hors_portee_gemv(monkeypatch):
    lin, x = _lin(), _x()
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX_PARTAGE", 0)
    ref, ch_ref = _chemins(lambda: lin(x))
    assert ch_ref.get("gemv") == 1 and "dequant" not in ch_ref
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX_PARTAGE", 16)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    hors, ch_hors = _chemins(lambda: lin(x))                     # hors portée : inchangé, au bit
    assert ch_hors.get("gemv") == 1 and torch.equal(hors, ref)
    with kernels.depaquetage_partage():
        dans, ch_dans = _chemins(lambda: lin(x))
    assert ch_dans.get("dequant") == 1 and "gemv" not in ch_dans
    rel = ((dans.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert 0 < rel <= 2 ** -6, rel                              # pas au bit, mais à l'ordre de l'arrondi bf16


def test_petit_m_reste_gemv_dans_la_portee(monkeypatch):
    lin, x = _lin(), _x(12)
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX_PARTAGE", 16)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    with kernels.depaquetage_partage():
        _, ch = _chemins(lambda: lin(x))
    assert "dequant" not in ch
