"""Pièce 156 (c) : fusions du décodage GDN livrées AU BIT (règle 9, `torch.equal`, jamais une tolérance).

F4 `ACVRAM_GDN_ETAT_EN_PLACE` : la récurrence fla écrit son état dans le tampon statique (h0 = ht) au lieu d'allouer
puis recopier. Bras cassant (prise) : `ht` omis → état jamais écrit → ROUGE dès le deuxième pas.
F5 `ACVRAM_GDN_RES_DIFFERE` : résidu différé (`decode_fixed_res`, `add_norm`) sur une pile de couches GDN contre
`decode_fixed` puis la norme finale. Bras cassant (prise) : `add_norm` remplacé par une somme non arrondie en bf16
avant la norme → ROUGE."""
import copy

import pytest
import torch

from acvram.engine import gdn as G
from acvram.engine import model as MO
from acvram.engine.couches import DecoderLayerGDN
from acvram.engine.gdn import GatedDeltaNet
from acvram.engine.layers import RMSNorm, add_norm

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau fla : carte requise"),
              pytest.mark.skipif(G._fla() is None, reason="flash-linear-attention absent")]
DEV = torch.device("cuda:0")
H, NK, NV, DK, DV, KER, B = 64, 2, 4, 16, 16, 4, 6


@pytest.fixture(autouse=True)
def _sans_grad(monkeypatch):
    # F1 (156 d, défaut) change les portes du seul bras « en place » : ± ulp, pas au bit. Ces tests jugent F4/F5/F2 AU
    # BIT, F1 à part (test_gdn_fusions_156d.py) — elle est donc coupée ici.
    monkeypatch.setattr(G, "_GDN_PORTES_NOYAU", False)
    with torch.no_grad():
        yield


def _lin(o, i, dtype, s=0.2):
    m = torch.nn.Linear(i, o, bias=False)
    m.weight.data.uniform_(-s, s)
    return m.to(dtype)


def _gdn(dtype=torch.float32, graine=20260924):
    torch.manual_seed(graine)
    kd, vd = NK * DK, NV * DV
    return GatedDeltaNet(
        qkv=_lin(2 * kd + vd, H, dtype), gate=_lin(vd, H, dtype), alpha=_lin(NV, H, dtype),
        beta=_lin(NV, H, dtype), out=_lin(H, vd, dtype), conv_weight=torch.randn(2 * kd + vd, KER) * 0.3,
        dt_bias=torch.rand(NV) - 0.5, a_log=torch.rand(NV) * 3 - 2,
        norm_weight=torch.ones(DV) + 0.1 * torch.randn(DV),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to(DEV)


def _etats_initiaux(couche, statics, graine=7):
    g = torch.Generator(device="cpu").manual_seed(graine)
    for st in statics:
        couche.static_load(st, (torch.randn(st["conv"].shape, generator=g).to(DEV),
                                torch.randn(st["S"].shape, generator=g).to(DEV) * 0.1))


def test_f4_etat_en_place_au_bit(monkeypatch):
    a = _gdn()
    b = copy.deepcopy(a)
    sa = [a.new_static(DEV) for _ in range(B)]
    sb = [b.new_static(DEV) for _ in range(B)]
    _etats_initiaux(a, sa); _etats_initiaux(b, sb)
    appels = []
    vrai = G._recurrence_en_place
    monkeypatch.setattr(G, "_recurrence_en_place", lambda *x: appels.append(1) or vrai(*x))
    torch.manual_seed(3)
    for pas in range(5):
        h = torch.randn(B, H, device=DEV)
        monkeypatch.setattr(G, "_GDN_ETAT_EN_PLACE", False)
        ya = a.decode_static_batch(h, sa)
        monkeypatch.setattr(G, "_GDN_ETAT_EN_PLACE", True)
        yb = b.decode_static_batch(h, sb)
        assert torch.equal(ya, yb), (pas, (ya - yb).abs().max().item())
        for i in range(B):
            assert torch.equal(sa[i]["S"], sb[i]["S"]), (pas, i)
            assert torch.equal(sa[i]["conv"], sb[i]["conv"]), (pas, i)
    assert len(appels) == 5, "le chemin en place doit avoir été pris"


def _pile(n=3):
    torch.manual_seed(11)
    couches = []
    for i in range(n):
        mlp = torch.nn.Sequential(_lin(2 * H, H, torch.bfloat16), torch.nn.SiLU(),
                                  _lin(H, 2 * H, torch.bfloat16)).to(DEV)
        norme = lambda: RMSNorm((torch.ones(H) + 0.1 * torch.randn(H)).to(torch.bfloat16).to(DEV), 1e-6)
        c = DecoderLayerGDN(i, _gdn(torch.bfloat16, 100 + i), mlp, norme(), norme(), DEV)
        c.statics = [c.linear_attn.new_static(DEV) for _ in range(B)]
        _etats_initiaux(c.linear_attn, c.statics, 50 + i)
        couches.append(c)
    return couches, RMSNorm((torch.ones(H) + 0.1 * torch.randn(H)).to(torch.bfloat16).to(DEV), 1e-6)


def test_f5_residu_differe_au_bit():
    pa, na = _pile()
    pb, nb = copy.deepcopy(pa), copy.deepcopy(na)
    torch.manual_seed(5)
    for pas in range(3):
        x0 = torch.randn(B, H, device=DEV).to(torch.bfloat16)
        x = x0.clone()
        for c in pa:
            x = c.decode_fixed(x, None, None, None, None, 0, None)
        ha = na(x)
        xb, delta = x0.clone(), None
        for c in pb:
            xb, delta = c.decode_fixed_res(xb, delta, None, None, None, None, 0, None)
        xb, hb = add_norm(xb, delta, nb)
        assert torch.equal(x, xb), (pas, (x.float() - xb.float()).abs().max().item())
        assert torch.equal(ha, hb), (pas, (ha.float() - hb.float()).abs().max().item())


def test_f5_le_predicat_admet_la_pile_gdn(monkeypatch):
    pile, norme = _pile(2)
    m = MO.ACVRamModel.__new__(MO.ACVRamModel)
    torch.nn.Module.__init__(m)
    m.layers = torch.nn.ModuleList(pile)
    m.norm = norme
    monkeypatch.setattr(MO, "_GDN_RES_DIFFERE", False)
    assert not m._res_differe()
    del m.__dict__["_res_ok"]
    monkeypatch.setattr(MO, "_GDN_RES_DIFFERE", True)
    assert m._res_differe()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("en_place", [False, True], ids=["copie", "F4"])
def test_f2_conv_fusee_au_bit(monkeypatch, dtype, en_place):
    """F2 seule puis F2 + F4, contre le chemin torch (cast, cat, conv1d, silu, repeat_interleave). NK ≠ NV : la
    répétition des têtes compte. Bras cassant (prise) : la FMA de la dernière prise remplacée par mul + add → ROUGE."""
    a = _gdn(dtype)
    b = copy.deepcopy(a)
    sa = [a.new_static(DEV) for _ in range(B)]
    sb = [b.new_static(DEV) for _ in range(B)]
    _etats_initiaux(a, sa); _etats_initiaux(b, sb)
    monkeypatch.setattr(G, "_GDN_ETAT_EN_PLACE", en_place)
    torch.manual_seed(9)
    for pas in range(5):
        h = torch.randn(B, H, device=DEV).to(dtype)
        monkeypatch.setattr(G, "_GDN_CONV_FUSEE", False)
        ya = a.decode_static_batch(h, sa)
        monkeypatch.setattr(G, "_GDN_CONV_FUSEE", True)
        yb = b.decode_static_batch(h, sb)
        assert torch.equal(ya, yb), (pas, (ya.float() - yb.float()).abs().max().item())
        for i in range(B):
            assert torch.equal(sa[i]["conv"], sb[i]["conv"]), (pas, i)
            assert torch.equal(sa[i]["S"], sb[i]["S"]), (pas, i)
