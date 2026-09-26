"""Pièce 245 : `GatedDeltaNet.coeur_lot` (préfill de plusieurs séquences : projections, convolution et out_proj par
séquence, portes + fla `cu_seqlens` + norme sur le lot) rend AU BIT la boucle `forward` séquence par séquence —
sorties, états de convolution et états récurrents, avec et sans état initial, longueurs mêlées (1, 17, 64, 65, 78, 130),
fp32 et bf16. Deux bras cassants dans le test : frontière fuyante (deux séquences fondues dans cu_seqlens) et états
initiaux échangés entre voisines — chacun DOIT rendre la comparaison fausse, sinon le test ne juge rien."""
import pytest
import torch

from acvram import regime
from acvram.engine import gdn as G
from acvram.engine.gdn import GatedDeltaNet

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="fla varlen : carte requise"),
              pytest.mark.skipif(G._fla() is None, reason="flash-linear-attention absent")]
H, NK, NV, DK, DV, KER = 64, 2, 4, 16, 16, 4


@pytest.fixture(autouse=True)
def _fla_sans_grad(monkeypatch):
    monkeypatch.setattr(G, "_GDN_VOIE", "fla")
    with torch.no_grad():
        yield


def _couche(dtype, seed=245) -> GatedDeltaNet:
    torch.manual_seed(seed)

    def lin(o, i):
        m = torch.nn.Linear(i, o, bias=False)
        m.weight.data.uniform_(-0.4, 0.4)
        return m.to(dtype)
    conv_dim = 2 * NK * DK + NV * DV
    return GatedDeltaNet(
        qkv=lin(conv_dim, H), gate=lin(NV * DV, H), alpha=lin(NV, H), beta=lin(NV, H), out=lin(H, NV * DV),
        conv_weight=(torch.randn(conv_dim, KER) * 0.3).to(dtype), dt_bias=torch.rand(NV) - 0.5,
        a_log=torch.rand(NV) * 3 - 2, norm_weight=torch.ones(DV) + 0.1 * torch.randn(DV),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to("cuda")


def _etats(couche, lens, dtype, avec):
    """États initiaux : None, ou ceux d'un premier préfill de chaque séquence (préfill par morceaux)."""
    if not avec:
        return [None] * len(lens)
    g = torch.Generator(device="cpu").manual_seed(7)
    return [couche(torch.randn(n0, H, generator=g).to("cuda", dtype), None)[1] for n0 in (40, 7, 64, 3, 66, 1, 33, 90)[:len(lens)]]


def _compare(couche, lens, dtype, avec, etats_lot=None):
    g = torch.Generator(device="cpu").manual_seed(sum(lens))
    h = torch.randn(sum(lens), H, generator=g).to("cuda", dtype)
    etats = _etats(couche, lens, dtype, avec)
    ref_y, ref_e, d = [], [], 0
    for n, e in zip(lens, etats):
        y, e2 = couche(h[d:d + n], e)
        ref_y.append(y); ref_e.append(e2); d += n
    ys, es = couche.coeur_lot(h, etats if etats_lot is None else etats_lot(etats), list(lens))
    ok = all(torch.equal(a, b) for a, b in zip(ref_y, ys))
    ok &= all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ref_e, es))
    return ok


LOTS = [(78,) * 8, (17, 64, 65, 130), (130, 2, 1, 78), (64, 64)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lens", LOTS)
@pytest.mark.parametrize("avec", [False, True])
def test_coeur_lot_au_bit_de_la_boucle(dtype, lens, avec):
    couche = _couche(dtype)
    if avec and 1 in lens:
        pytest.skip("t = 1 avec état = décodage, refusé en amont (couches.py)")
    assert _compare(couche, lens, dtype, avec)


def test_cassant_frontiere_fuyante(monkeypatch):
    """Les deux premières séquences fondues dans cu_seqlens : la causalité passe la frontière → la comparaison DOIT échouer."""
    couche = _couche(torch.float32)
    vraie = G._fla()

    def fuyante(q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens=None):
        if cu_seqlens is None:                      # boucle de référence (`_coeur`) : fla intact
            return vraie[0](q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
        cu = torch.cat([cu_seqlens[:1], cu_seqlens[2:]])
        o, s = vraie[0](q, k, v, g=g, beta=beta, initial_state=initial_state[1:].contiguous(),
                        output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        cu_seqlens=cu)
        return o, torch.cat([s[:1], s])
    monkeypatch.setattr(G, "_fla", lambda: (fuyante, vraie[1]))
    assert not _compare(couche, (17, 64, 65, 130), torch.float32, False)


def test_cassant_etats_echanges():
    couche = _couche(torch.float32)
    assert not _compare(couche, (17, 64, 65, 130), torch.float32, True, etats_lot=lambda e: e[1:] + e[:1])


def test_variable_dans_la_table_de_regime():
    assert any(v.nom == "GDN_COEUR_LOT" for v in regime.VARIABLES)
