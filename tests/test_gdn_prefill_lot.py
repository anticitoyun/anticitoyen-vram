"""Pièce 150 bis : préfill Gated DeltaNet du LOT (`forward_lot`, projections en un
appel) contre b appels `forward` séquence par séquence (règle 9). Bras cassant : le
lot passé comme UNE séquence (état et convolution qui traversent la frontière)
doit rendre rouge — sinon le contrôle ne verrait pas la faute qu'il garde."""
import pytest
import torch

from acvram.engine.gdn import GatedDeltaNet, gdn_available

pytestmark = pytest.mark.skipif(not gdn_available(), reason="transformers/qwen3_next absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEUIL = 1e-4
H, NK, NV, DK, DV, KER = 48, 2, 4, 8, 8, 4


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin(o, i, s=0.3):
    m = torch.nn.Linear(i, o, bias=False)
    m.weight.data.uniform_(-s, s)
    return m.float()


def _gdn():
    torch.manual_seed(20260924)
    kd, vd = NK * DK, NV * DV
    return GatedDeltaNet(
        qkv=_lin(2 * kd + vd, H), gate=_lin(vd, H), alpha=_lin(NV, H), beta=_lin(NV, H),
        out=_lin(H, vd), conv_weight=torch.randn(2 * kd + vd, KER) * 0.3,
        dt_bias=torch.rand(NV) - 0.5, a_log=torch.rand(NV) * 3 - 2,
        norm_weight=torch.ones(DV) + 0.1 * torch.randn(DV),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to(DEV)


def _ecart(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _cas():
    """Trois séquences de longueurs différentes, la deuxième reprend un état (tranche
    suivante d'un préfill découpé ou préfixe servi)."""
    couche = _gdn()
    torch.manual_seed(7)
    _, etat_pris = couche(torch.randn(9, H, device=DEV), None)
    lens = [13, 6, 21]
    etats = [None, etat_pris, None]
    x = torch.randn(sum(lens), H, device=DEV)
    return couche, x, lens, etats


def test_le_lot_vaut_b_appels():
    couche, x, lens, etats = _cas()
    y, e_new = couche.forward_lot(x, etats, lens)
    d = 0
    for s, ql in enumerate(lens):
        y_ref, e_ref = couche(x[d:d + ql], etats[s])
        assert _ecart(y[d:d + ql], y_ref) < SEUIL, (s, _ecart(y[d:d + ql], y_ref))
        for j in range(2):
            assert _ecart(e_new[s][j], e_ref[j]) < SEUIL, (s, j)
        d += ql


def test_bras_cassant_frontiere_traversee():
    couche, x, lens, etats = _cas()
    y, _ = couche.forward_lot(x, etats, lens)
    y_faux, _ = couche(x, None)                  # le lot comme une seule séquence
    fin0 = lens[0]
    assert _ecart(y_faux[fin0:], y[fin0:]) > 1e-2
