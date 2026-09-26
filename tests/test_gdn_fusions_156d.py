"""Pièce 156 (d) (scellé `revue/poste5-piece156d-scelle-24-09.md`) : F1, F3 (± ulp, bornés ici, jugés par la KL de la
prise) et F6 (AU BIT, `torch.equal`).

F1 `ACVRAM_GDN_PORTES_NOYAU` : portes dans le noyau fla (voie F4). Bras cassant (prise) : `dt_bias` omis → ROUGE.
F3 `ACVRAM_GDN_NORME_FUSEE` : norme gated en un noyau Triton, ≤ 1 ulp bf16. Bras cassant : poids de la norme oublié.
F6 `ACVRAM_NORME_REGISTRES` : `rmsnorm_bf16_reg` contre `rmsnorm_bf16`, au bit. Bras cassant : somme d'un fil inversée."""
import copy

import pytest
import torch

from acvram import kernels
from acvram.engine import gdn as G
from acvram.engine import layers as L
from acvram.engine.gdn import GatedDeltaNet
from acvram.engine.layers import RMSNorm, add_norm

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise"),
              pytest.mark.skipif(G._fla() is None, reason="flash-linear-attention absent")]
DEV = torch.device("cuda:0")
H, NK, NV, DK, DV, KER, B = 64, 2, 4, 16, 16, 4, 6


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin(o, i, dtype, s=0.2):
    m = torch.nn.Linear(i, o, bias=False)
    m.weight.data.uniform_(-s, s)
    return m.to(dtype)


def _gdn(dtype=torch.float32, dv=DV, graine=20260924):
    torch.manual_seed(graine)
    kd, vd = NK * DK, NV * dv
    return GatedDeltaNet(
        qkv=_lin(2 * kd + vd, H, dtype), gate=_lin(vd, H, dtype), alpha=_lin(NV, H, dtype),
        beta=_lin(NV, H, dtype), out=_lin(H, vd, dtype), conv_weight=torch.randn(2 * kd + vd, KER) * 0.3,
        dt_bias=torch.rand(NV) - 0.5, a_log=torch.rand(NV) * 3 - 2,
        norm_weight=torch.ones(dv) + 0.1 * torch.randn(dv),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=dv).to(DEV)


def _etats_initiaux(couche, statics, graine=7):
    g = torch.Generator(device="cpu").manual_seed(graine)
    for st in statics:
        couche.static_load(st, (torch.randn(st["conv"].shape, generator=g).to(DEV),
                                torch.randn(st["S"].shape, generator=g).to(DEV) * 0.1))


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_f1_portes_dans_le_noyau(monkeypatch, dtype):
    """Voie F4 + F2 (défaut) contre la même voie, portes dans le noyau : sorties et états à ± ulp fp32 sur 5 pas
    (bf16 : la sortie d'out_proj peut basculer d'un ulp bf16)."""
    a = _gdn(dtype)
    b = copy.deepcopy(a)
    sa = [a.new_static(DEV) for _ in range(B)]
    sb = [b.new_static(DEV) for _ in range(B)]
    _etats_initiaux(a, sa); _etats_initiaux(b, sb)
    monkeypatch.setattr(G, "_GDN_ETAT_EN_PLACE", True)
    monkeypatch.setattr(G, "_GDN_CONV_FUSEE", True)
    appels = []
    vrai = G._recurrence_en_place
    monkeypatch.setattr(G, "_recurrence_en_place", lambda *x: appels.append(len(x)) or vrai(*x))
    tol_y = 2.0 ** -16 if dtype == torch.float32 else 2.0 ** -7
    torch.manual_seed(3)
    for pas in range(5):
        h = torch.randn(B, H, device=DEV).to(dtype)
        monkeypatch.setattr(G, "_GDN_PORTES_NOYAU", False)
        ya = a.decode_static_batch(h, sa)
        monkeypatch.setattr(G, "_GDN_PORTES_NOYAU", True)
        yb = b.decode_static_batch(h, sb)
        assert _rel(yb, ya) <= tol_y, (pas, _rel(yb, ya))
        for i in range(B):
            assert _rel(sb[i]["S"], sa[i]["S"]) <= 2.0 ** -16, (pas, i, _rel(sb[i]["S"], sa[i]["S"]))
            assert torch.equal(sa[i]["conv"], sb[i]["conv"]), (pas, i)
    assert appels == [6, 8] * 5, appels        # 8 arguments = A_log et dt_bias passés : F1 prise


def test_f1_sans_voie_f2_retombe_sur_les_portes_torch(monkeypatch):
    """F1 sans F2 : `_lot_projete` rend des portes déjà calculées ; le noyau ne doit pas les recalculer."""
    a = _gdn()
    b = copy.deepcopy(a)
    sa, sb = [a.new_static(DEV) for _ in range(B)], [b.new_static(DEV) for _ in range(B)]
    _etats_initiaux(a, sa); _etats_initiaux(b, sb)
    monkeypatch.setattr(G, "_GDN_ETAT_EN_PLACE", True)
    monkeypatch.setattr(G, "_GDN_CONV_FUSEE", False)
    h = torch.randn(B, H, device=DEV)
    monkeypatch.setattr(G, "_GDN_PORTES_NOYAU", False)
    ya = a.decode_static_batch(h, sa)
    monkeypatch.setattr(G, "_GDN_PORTES_NOYAU", True)
    yb = b.decode_static_batch(h, sb)
    assert torch.equal(ya, yb)


def _ulp_bf16(a, b):
    """Écart en ulp bf16 élément par élément (mêmes signes attendus)."""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    return (ia - ib).abs()


def test_f3_norme_gated_a_un_ulp_bf16(monkeypatch):
    c = _gdn(dv=128)
    torch.manual_seed(5)
    x = torch.randn(384, 128, device=DEV) * 3
    z = torch.randn(384, 128, device=DEV) * 2
    monkeypatch.setattr(G, "_GDN_NORME_FUSEE", False)
    ref = c._norm_gated(x, z, torch.bfloat16).to(torch.bfloat16)
    monkeypatch.setattr(G, "_GDN_NORME_FUSEE", True)
    y = c._norm_gated(x, z, torch.bfloat16)
    assert y.dtype == torch.bfloat16
    assert int(_ulp_bf16(y, ref).max()) <= 1
    # sortie non bf16 demandée : F3 n'est pas prise (le noyau n'écrit que du bf16)
    assert c._norm_gated(x, z, torch.float32).dtype == torch.float32


def test_f3_prise_au_decodage_du_lot(monkeypatch):
    a = _gdn(torch.bfloat16)
    b = copy.deepcopy(a)
    sa, sb = [a.new_static(DEV) for _ in range(B)], [b.new_static(DEV) for _ in range(B)]
    _etats_initiaux(a, sa); _etats_initiaux(b, sb)
    from acvram.engine import gdn_norme
    appels = []
    vrai = gdn_norme.norme_gated
    monkeypatch.setattr(gdn_norme, "norme_gated", lambda *x: appels.append(1) or vrai(*x))
    h = torch.randn(B, H, device=DEV).to(torch.bfloat16)
    monkeypatch.setattr(G, "_GDN_NORME_FUSEE", False)
    ya = a.decode_static_batch(h, sa)
    monkeypatch.setattr(G, "_GDN_NORME_FUSEE", True)
    yb = b.decode_static_batch(h, sb)
    assert appels == [1]
    assert _rel(yb, ya) <= 2.0 ** -6, _rel(yb, ya)


def _ext():
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "rmsnorm_bf16_reg"):
        pytest.skip("extension sans rmsnorm_bf16_reg")
    return ext


@pytest.mark.parametrize("Hn", [128, 256, 1000, 2048, 3000, 5120, 8192])
@pytest.mark.parametrize("R", [1, 8, 48])
def test_f6_rmsnorm_registres_au_bit(Hn, R):
    ext = _ext()
    g = torch.Generator(device="cpu").manual_seed(Hn * 100 + R)
    x = (torch.randn(R, Hn, generator=g) * 2).to(torch.bfloat16).to(DEV)
    r = (torch.randn(R, Hn, generator=g) * 4).to(torch.bfloat16).to(DEV)
    w = (1 + 0.2 * torch.randn(Hn, generator=g)).to(torch.bfloat16).to(DEV)
    assert torch.equal(ext.rmsnorm_bf16_reg(x, w, 1e-6)[0], ext.rmsnorm_bf16(x, w, 1e-6)[0])
    for mult in (1.0, 0.5):
        ha, xa = ext.rmsnorm_bf16(x, w, 1e-6, r, mult)
        hb, xb = ext.rmsnorm_bf16_reg(x, w, 1e-6, r, mult)
        assert torch.equal(xa, xb) and torch.equal(ha, hb), (Hn, R, mult)


def test_f6_prise_par_rmsnorm_et_add_norm(monkeypatch):
    ext = _ext()
    appels = []
    vrai = ext.rmsnorm_bf16_reg

    class Espion:
        def __getattr__(self, n):
            if n == "rmsnorm_bf16_reg":
                return lambda *a: appels.append(1) or vrai(*a)
            return getattr(ext, n)
    monkeypatch.setattr(kernels, "get_extension", lambda: Espion())
    norme = RMSNorm(torch.ones(5120, dtype=torch.bfloat16, device=DEV) * 1.1, 1e-6)
    x = torch.randn(8, 5120, device=DEV).to(torch.bfloat16)
    r = torch.randn(8, 5120, device=DEV).to(torch.bfloat16)
    monkeypatch.setattr(L, "_NORME_REGISTRES", False)
    ha, (xa, na) = norme(x), add_norm(r, x, norme)
    monkeypatch.setattr(L, "_NORME_REGISTRES", True)
    hb, (xb, nb) = norme(x), add_norm(r, x, norme)
    assert appels == [1, 1]
    assert torch.equal(ha, hb) and torch.equal(xa, xb) and torch.equal(na, nb)
