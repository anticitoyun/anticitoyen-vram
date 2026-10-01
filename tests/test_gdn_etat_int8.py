"""État GDN INT8 par fenêtre (LeapQuant, opt-in ``ACVRAM_ETAT_GDN=int8``) : la sortie CHANGE, donc l'équivalence se
juge à tolérances NOMMÉES, écrites avant la mesure (revue/poste5-leap-int8-scelle-01-10.md), en trois étages :

1. l'ALGÈBRE de la fenêtre (tampon fp32, aucune quantification dans l'horizon) contre la récurrence exacte :
   écart relatif ≤ ``TOL_ALGEBRE`` = 1e-5 sur 40 pas — casse sur toute faute d'indice, de décroissance ou de GQA ;
2. la QUANTIFICATION (tampon fp16, INT8, P = 16) : écart relatif de la sortie ≤ ``TOL_INT8`` = 3e-2 à chaque pas sur
   256 pas, et SANS dérive (moyenne des 64 derniers pas ≤ 2 × celle des pas 16-80) ;
3. le contrôle qui peut rendre faux : la même quantification PAR PAS (P = 1, l'erreur à chaque jeton — le cas que
   l'article montre catastrophique) doit faire PIRE que la fenêtre ; sinon l'instrument ne voit pas ce qu'il juge.
Sur carte, les noyaux Triton se jugent contre cette référence (tests suivants)."""
import pytest
import torch

from acvram.engine import gdn_etat_int8 as E

TOL_ALGEBRE, TOL_INT8 = 1e-5, 3e-2
N, HV, H, K, V = 2, 6, 2, 32, 32


def _entrees(pas, graine, dev="cpu"):
    g_ = torch.Generator().manual_seed(graine)
    q, k = torch.randn(N, H, K, generator=g_), torch.randn(N, H, K, generator=g_)
    v = torch.randn(N, HV, V, generator=g_)
    g = -torch.rand(N, HV, generator=g_) * 0.1                     # α ∈ (0,90 ; 1], la plage servie
    beta = torch.rand(N, HV, generator=g_)
    return [t.to(dev) for t in (q, k, v, g, beta)]


def _ecart(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def _trajet(fenetre, pas, tampon, graine=3, S0=None, dev="cpu", monkeypatch=None):
    monkeypatch.setattr(E, "P", fenetre)
    et = E.nouvel_etat(N, HV, H, K, V, device=dev, tampon=tampon)
    if S0 is not None:
        E.geler(et, S0.clone())
    S = E.etat_fp32(et)                                             # le témoin part du MÊME état (déquantifié)
    ecarts = []
    for t in range(pas):
        x = _entrees(t, 1000 * graine + t, dev)
        o = E.pas_reference(et, *x)
        o_ref = E.pas_fp32(S, *x)
        ecarts.append(_ecart(o, o_ref))
        if fenetre > pas:                                           # algèbre seule : l'état reconstruit aussi
            assert _ecart(E.etat_fp32(et), S) <= TOL_ALGEBRE, f"état ≠ témoin au pas {t}"
    return ecarts


def test_algebre_de_la_fenetre(monkeypatch):
    S0 = torch.randn(N, HV, K, V, generator=torch.Generator().manual_seed(5)) * 0.3
    e = _trajet(64, 40, torch.float32, S0=S0, monkeypatch=monkeypatch)
    assert max(e) <= TOL_ALGEBRE, max(e)


def test_quantification_par_fenetre_sans_derive(monkeypatch):
    e = _trajet(16, 256, torch.float16, monkeypatch=monkeypatch)
    assert max(e) <= TOL_INT8, max(e)
    assert sum(e[-64:]) / 64 <= 2 * sum(e[16:80]) / 64, (sum(e[-64:]) / 64, sum(e[16:80]) / 64)


def test_controle_par_pas_fait_pire(monkeypatch):
    fen = sum(_trajet(16, 128, torch.float16, monkeypatch=monkeypatch)[-64:])
    par_pas = sum(_trajet(1, 128, torch.float16, monkeypatch=monkeypatch)[-64:])
    assert par_pas > fen, (par_pas, fen)


def test_geler_puis_reconstruire(monkeypatch):
    """Bord seul : S → (Z, B, C, compensateurs) → S ; écart relatif ≤ 1/127 (pas d'un INT8 symétrique)."""
    monkeypatch.setattr(E, "P", 16)
    et = E.nouvel_etat(N, HV, H, K, V)
    S = torch.randn(N, HV, K, V, generator=torch.Generator().manual_seed(9))
    S[:, :, 3] *= 40                                                # une ligne aberrante, comme l'article les décrit
    E.geler(et, S)
    assert _ecart(E.etat_fp32(et), S) <= 1 / 127


# --- sur carte : noyaux Triton contre la référence, dimensions de Qwen3.8 ------------------------------------------
TOL_NOYAU = 1e-3          # même algèbre, autre ordre de sommation, arrondis INT8 à égalité possibles
carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyaux Triton : carte requise")


def _reel(graine, n=12, hv=48, h=16, porte=False):
    g_ = torch.Generator().manual_seed(graine)
    d = "cuda:0"
    q, k = torch.randn(n, 1, h, 128, generator=g_).to(d), torch.randn(n, 1, h, 128, generator=g_).to(d)
    v = torch.randn(n, 1, hv, 128, generator=g_).to(d)
    if porte:
        return q, k, v, torch.randn(n, 1, hv, generator=g_).to(d), torch.randn(n, 1, hv, generator=g_).to(d)
    return q, k, v, (-torch.rand(n, 1, hv, generator=g_) * 0.1).to(d), torch.rand(n, 1, hv, generator=g_).to(d)


@carte
@pytest.mark.parametrize("porte", [False, True])
def test_noyaux_contre_reference(porte):
    import torch.nn.functional as F
    n, hv, h = 12, 48, 16
    A, dt = torch.rand(hv, device="cuda:0") - 1.0, torch.rand(hv, device="cuda:0") - 0.5
    et_r = E.nouvel_etat(n, hv, h, device="cuda:0")
    E.geler(et_r, torch.randn(n, hv, 128, 128, generator=torch.Generator().manual_seed(1)).to("cuda:0") * 0.3)
    et_t = {c: t.clone() for c, t in et_r.items()}
    S = E.etat_fp32(et_r)
    for t in range(40):                                           # deux bords de fenêtre (P = 16)
        q, k, v, g, b = _reel(100 + t, porte=porte)
        o_t = E.pas(et_t, q, k, v, g, b, A if porte else None, dt if porte else None)[:, 0]
        if porte:
            g, b = -A.exp() * F.softplus(g + dt), b.sigmoid()
        args = (q[:, 0], k[:, 0], v[:, 0], g[:, 0], b[:, 0])
        o_r = E.pas_reference(et_r, *args)
        o_f = E.pas_fp32(S, *args)
        assert _ecart(o_t, o_r) <= TOL_NOYAU, (t, _ecart(o_t, o_r))
        assert torch.equal(et_t["n"], et_r["n"]), t
        assert _ecart(E.etat_fp32(et_t), E.etat_fp32(et_r)) <= TOL_NOYAU, t
        assert _ecart(o_t, o_f) <= TOL_INT8, (t, _ecart(o_t, o_f))


@carte
def test_couche_gdn_int8_contre_fp32_et_export_exact(monkeypatch):
    """La couche entière (projections, conv, norme, out_proj) : voie int8 à TOL_INT8 de la voie fp32 sur 40 pas ;
    puis export → chargement dans un autre créneau → pas suivant AU BIT (l'aller-retour ne requantifie pas)."""
    import copy
    from acvram.engine import gdn as G
    from test_gdn_fusions_au_bit import _gdn, DEV
    torch.manual_seed(0)
    a = _gdn()
    b = copy.deepcopy(a)
    st_a = [a.new_static(DEV) for _ in range(4)]
    monkeypatch.setattr(G, "_ETAT_GDN", "int8")
    st_b = [b.new_static(DEV) for _ in range(4)]
    gen = torch.Generator().manual_seed(3)
    for sa, sb in zip(st_a, st_b):
        e = (torch.randn(sa["conv"].shape, generator=gen).to(DEV), torch.randn(sa["S"].shape, generator=gen).to(DEV) * .1)
        G.GatedDeltaNet.static_load(sa, e); G.GatedDeltaNet.static_load(sb, e)
    with torch.no_grad():
        for t in range(40):
            h = torch.randn(4, 64, generator=gen).to(DEV)
            ya, yb = a.decode_static_batch(h, st_a), b.decode_static_batch(h, st_b)
            assert _ecart(yb.float(), ya.float()) <= TOL_INT8, (t, _ecart(yb.float(), ya.float()))
        exp = G.GatedDeltaNet.static_export(st_b[1])
        autre = b.new_static(DEV)
        G.GatedDeltaNet.static_load(autre, exp)
        h = torch.randn(1, 64, generator=gen).to(DEV)
        y1 = b.decode_static(h, st_b[1])
        y2 = b.decode_static(h, autre)
        assert torch.equal(y1, y2)
