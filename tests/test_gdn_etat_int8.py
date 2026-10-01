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


# --- noyaux contre référence : bornes DÉRIVÉES, pas resynchronisés ------------------------------------------------
# (01/10, chef : la « TOL_NOYAU = 1e-3 » d'avant était posée à la main ; rouge sur carte, 1,24e-3 au 2e gel.) Deux
# gels corrects ne rendent pas les mêmes codes : à sec, K = V = 128, l'écart d'état entre la forme close et la récurrence
# séquentielle croît puis SATURE à ~1,4 ε (√2 ε : deux quantifications indépendantes ; ε ≈ 4,3e-3 l'erreur d'un gel),
# scratchpad/poste5-int8-01-10/derive_deux_quantifs.txt. Comparer deux trajectoires libres ne juge donc pas le noyau.
# Ici chaque pas part du MÊME état compressé (recopié), et chaque écart a sa borne d'analyse d'erreur directe :
#   sortie et enregistrement w : γ_n · (décomposition en VALEURS ABSOLUES de l'état : résidu, compensateurs, tampon —
#     ils s'annulent entre eux, |S| les sous-estimerait), n = 2K + 2P + R + 16 (sommes sur K et sur le tampon,
#     compensateurs, transcendantes et normes L2 à quelques ulp), u = 2⁻²⁴ ; w stocké en fp16 : + 2⁻¹⁰ |w| (deux fp32
#     à moins d'une ulp fp16 d'écart peuvent s'arrondir aux deux voisins) ;
#   état après un gel (norme de Frobenius par tête) : chacun est à ≤ ‖C‖·‖B‖/254 de l'état d'avant gel (demi-marche
#     de l'INT8 symétrique, les compensateurs étant soustraits avant quantification), et les deux états d'avant gel
#     ne diffèrent que d'arrondis fp32 : borne (‖C_T‖‖B_T‖ + ‖C_R‖‖B_R‖)/254 + γ_n ‖décomposition‖.
# La dérive et la justesse se jugent à part, en roue libre contre la récurrence exacte (TOL_INT8, sans dérive).
U32, D16 = 2.0 ** -24, 2.0 ** -10


def _gamma(K, P_, R_):
    n = 2 * K + 2 * P_ + R_ + 16
    return n * U32 / (1 - n * U32)


def _absolu(et):
    """Majorant élément par élément de |S| sans annulation : e^{Pn}(|résidu| + Σ|k̃||ũ|ᵀ) + Σ_j poids_j |k_j||w_j|ᵀ."""
    N_, HV_, K_, V_ = et["Z"].shape
    gqa = HV_ // et["bk"].shape[1]
    S0 = (et["C"][..., :, None] * et["Z"].float().abs() / 127.0 * et["B"][..., None, :]
          + torch.einsum("nhrk,nhrv->nhkv", et["kc"].float().abs(), et["uc"].float().abs()))
    n = et["n"].long()
    j = torch.arange(E.P, device=S0.device)
    actif = j[None, None, :] < n[..., None]
    bg = torch.where(actif, et["bg"], torch.zeros_like(et["bg"]))
    pref = torch.cumsum(bg, -1)
    Pn = pref.gather(-1, (n - 1).clamp(min=0)[..., None])[..., 0] * (n > 0)
    poids = torch.exp(Pn[..., None] - pref) * actif
    bk = et["bk"].float().abs().repeat_interleave(gqa, 1)
    return torch.exp(Pn)[..., None, None] * S0 + torch.einsum("nhp,nhpk,nhpv->nhkv", poids, bk, et["bw"].float().abs())


def _bornes_pas(et, q, k, v, g, beta, dk=None):
    """Bornes élément par élément de |o_T − o_R| et |w_T − w_R| pour un pas parti de ``et`` (formes de la référence).
    ``dk`` [N, H, K] : écart MESURÉ des clés fp16 du pas (lu dans les deux enregistrements `bk`). Les deux côtés
    normalisent k dans un ordre de somme différent puis l'arrondissent en fp16 : quelques éléments basculent d'une
    ulp fp16 (à sec, deux ordres de somme : 1,0e-4 des éléments, 102 sur 40 pas à b=12) — trou de la première
    dérivation, rouge sur carte le 01/10 07:06 (revue/poste5-leap-int8-prise-01-10.md)."""
    N_, HV_, V_ = v.shape
    H_, K_ = q.shape[1], q.shape[2]
    gqa, gam = HV_ // H_, _gamma(K_, E.P, E.R)
    qn = (q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6) * K_ ** -0.5).repeat_interleave(gqa, 1)
    kn = (k.float() / torch.sqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(et["bk"].dtype).float()
    kn = kn.repeat_interleave(gqa, 1)
    M = _absolu(et)
    a, b = torch.exp(g.float())[..., None], beta.float()[..., None]
    mq, mk = torch.einsum("nhkv,nhk->nhv", M, qn.abs()), torch.einsum("nhkv,nhk->nhv", M, kn.abs())
    bw = gam * b * (v.float().abs() + a * mk)
    w = b * (v.float() - a * E._s_t(et, kn, gqa))
    kq = (kn * qn).sum(-1, keepdim=True).abs()
    if dk is not None:                                  # bascules fp16 de la clé du pas : s_k, w et k·q bougent
        dkh = dk.float().repeat_interleave(gqa, 1)
        ds = b * a * torch.einsum("nhkv,nhk->nhv", M, dkh)
        dkq = (dkh * qn.abs()).sum(-1, keepdim=True)
        bw = bw + ds
    else:
        ds, dkq = torch.zeros_like(bw), torch.zeros_like(kq)
    bo = gam * (a * mq + kq * w.abs()) + kq * bw + dkq * (w.abs() + ds)
    return bo, bw + D16 * w.abs()


def _q_gel(et):
    return et["C"].norm(dim=-1) * et["B"].norm(dim=-1) / 254.0          # [N, HV]


def verifier_resynchro(pas_t, et0, entrees, pas_n):
    """``pas_t(et, *x)`` (le candidat) contre `pas_reference`, chaque pas parti du même état compressé. Rend le
    nombre de gels traversés ; lève AssertionError à la première borne violée."""
    et_t, gels = {c: x.clone() for c, x in et0.items()}, 0
    stats = {"tetes": 0, "tetes_bascule": 0, "expliquees": 0}
    gam = _gamma(et0["Z"].shape[2], E.P, E.R)
    for t in range(pas_n):
        x = entrees(t)
        et_r = {c: y.clone() for c, y in et_t.items()}
        et_avant = {c: y.clone() for c, y in et_t.items()}
        n_avant = et_r["n"].clone()
        o_r = E.pas_reference(et_r, *x)
        o_t = pas_t(et_t, *x)
        nk = n_avant.view(n_avant.shape[0], et_r["bk"].shape[1], -1)[..., 0].long()
        ik = nk[..., None, None].expand(*nk.shape, 1, et_r["bk"].shape[-1])
        dk = (et_t["bk"].gather(2, ik) - et_r["bk"].gather(2, ik)).float().abs()[:, :, 0]      # [N, H, K]
        # Têtes SANS bascule de clé : borne STRICTE (sans terme Δk) — un seul hors-borne = bogue du noyau (C2′).
        # Têtes AVEC bascule : comptées à part comme « expliquées » (borne avec le Δk mesuré), jamais « tenues »
        # (chef 01/10) ; un hors-borne même là reste un rouge.
        bo0, bw0 = _bornes_pas(et_avant, *x)
        bo, bw = _bornes_pas(et_avant, *x, dk=dk)
        bascule = (dk.sum(-1).repeat_interleave(o_r.shape[1] // dk.shape[1], 1) > 0)[..., None].expand_as(o_r)
        d = (o_t - o_r).abs()
        rouge_strict = (d > bo0) & ~bascule
        rouge_bascule = (d > bo) & bascule
        stats["tetes_bascule"] += int(bascule[..., 0].sum()); stats["tetes"] += int(bascule[..., 0].numel())
        stats["expliquees"] += int(((d > bo0) & bascule).sum())
        if bool(rouge_strict.any() or rouge_bascule.any()):
            raise AssertionError(f"pas {t} : {int(rouge_strict.sum())} sorties hors borne STRICTE sur des têtes SANS "
                                 f"bascule (bogue), {int(rouge_bascule.sum())} hors borne même avec Δk ; rapport max "
                                 f"strict {float(torch.where(bascule, torch.zeros_like(d), d / bo0.clamp(min=1e-30)).max()):.3g}")
        bw = torch.where(bascule[..., :1].expand_as(bw), bw, bw0)
        gel = bool((et_r["n"] == 0).all() and (n_avant == E.P - 1).all())
        if not gel:
            assert torch.equal(et_t["n"], et_r["n"]), t
            idx = n_avant.long()[..., None, None].expand(*n_avant.shape, 1, et_r["bw"].shape[-1])
            dw = (et_t["bw"].gather(2, idx) - et_r["bw"].gather(2, idx)).float().abs()[:, :, 0]
            assert (dw <= bw).all(), (t, "enregistrement w", float((dw / bw.clamp(min=1e-30)).max()))
        else:
            gels += 1
            assert torch.equal(et_t["n"], et_r["n"]), t
            d = (E.etat_fp32(et_t) - E.etat_fp32(et_r)).flatten(2).norm(dim=-1)
            borne = _q_gel(et_t) + _q_gel(et_r) + gam * _absolu(et_r).flatten(2).norm(dim=-1)
            assert (d <= borne).all(), (t, "gel", float((d / borne).max()))
        et_t = {c: y.clone() for c, y in et_t.items()}
    print(f"RESUME resynchro : {gels} gels ; têtes×pas {stats['tetes']}, dont avec bascule de clé "
          f"{stats['tetes_bascule']} (jugées à part) ; sorties hors borne stricte expliquées par une bascule "
          f"{stats['expliquees']} — non comptées comme tenues", flush=True)
    return gels


def _entrees_k128(dev="cpu", n=2, hv=4, h=2):
    def f(t):
        g_ = torch.Generator().manual_seed(500 + t)
        q, k = torch.randn(n, h, 128, generator=g_), torch.randn(n, h, 128, generator=g_)
        v = torch.randn(n, hv, 128, generator=g_)
        return [z.to(dev) for z in (q, k, v, -torch.rand(n, hv, generator=g_) * 0.1, torch.rand(n, hv, generator=g_))]
    return f


def _depart(n=2, hv=4, h=2, dev="cpu"):
    et = E.nouvel_etat(n, hv, h, device=dev)
    E.geler(et, (torch.randn(n, hv, 128, 128, generator=torch.Generator().manual_seed(1)) * 0.3).to(dev))
    return et


def _variante_sequentielle(et, *x):
    """Une implémentation CORRECTE mais d'un autre ordre (celui du noyau de bord) : doit tenir les bornes."""
    efp = E.etat_fp32

    def seq(e):
        if int(e["n"].max()) != E.P:
            return efp(e)
        S = efp({**e, "n": torch.zeros_like(e["n"])})
        gqa = e["bw"].shape[1] // e["bk"].shape[1]
        for j in range(E.P):
            kj = e["bk"][:, :, j].float().repeat_interleave(gqa, 1)
            S = S * torch.exp(e["bg"][:, :, j])[..., None, None] + kj[..., :, None] * e["bw"][:, :, j].float()[..., None, :]
        return S
    E.etat_fp32 = seq
    try:
        return E.pas_reference(et, *x)
    finally:
        E.etat_fp32 = efp


def test_bornes_tiennent_pour_une_variante_correcte(monkeypatch):
    monkeypatch.setattr(E, "P", 16)
    assert verifier_resynchro(_variante_sequentielle, _depart(), _entrees_k128(), 40) == 2


def test_bornes_rejettent_une_faute_d_echelle(monkeypatch):
    """Le contrôle qui peut rendre faux : lire Z/126 au lieu de Z/127 (0,8 % sur la part INT8) doit casser."""
    monkeypatch.setattr(E, "P", 16)

    def fautive(et, *x):
        B = et["B"].clone(); et["B"].mul_(127.0 / 126.0)
        try:
            return E.pas_reference(et, *x)
        finally:
            if int(et["n"].max()) != 0:
                et["B"].copy_(B)
    with pytest.raises(AssertionError):
        verifier_resynchro(fautive, _depart(), _entrees_k128(), 40)


carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyaux Triton : carte requise")


@carte
@pytest.mark.parametrize("porte", [False, True])
def test_noyaux_contre_reference(porte, monkeypatch):
    import torch.nn.functional as F
    monkeypatch.setattr(E, "P", 16)
    n, hv, h, d = 12, 48, 16, "cuda:0"
    A, dt = torch.rand(hv, device=d) - 1.0, torch.rand(hv, device=d) - 0.5
    brut = _entrees_k128(d, n, hv, h)

    def entrees(t):
        q, k, v, g, b = brut(t)
        if porte:                                       # portes brutes au noyau, portes calculées à la référence
            g_brut, b_brut = torch.randn_like(g), torch.randn_like(b)
            g, b = -A.exp() * F.softplus(g_brut + dt), b_brut.sigmoid()
            entrees.brutes[t] = (g_brut, b_brut)
        return q, k, v, g, b
    entrees.brutes = {}

    def triton_pas(et, q, k, v, g, b):
        t = len(triton_pas.vu); triton_pas.vu.append(t)
        if porte:
            g, b = entrees.brutes[t]
            return E.pas(et, q[:, None], k[:, None], v[:, None], g[:, None], b[:, None], A, dt)[:, 0]
        return E.pas(et, q[:, None], k[:, None], v[:, None], g[:, None], b[:, None])[:, 0]
    triton_pas.vu = []
    assert verifier_resynchro(triton_pas, _depart(n, hv, h, d), entrees, 40) == 2


@carte
def test_noyaux_roue_libre_contre_exact(monkeypatch):
    """Justesse et dérive du chemin Triton seul, 256 pas, contre la récurrence exacte (mêmes critères que la référence)."""
    monkeypatch.setattr(E, "P", 16)
    et = _depart(12, 48, 16, "cuda:0")
    S = E.etat_fp32(et)
    f, ecarts = _entrees_k128("cuda:0", 12, 48, 16), []
    for t in range(256):
        q, k, v, g, b = f(t)
        o = E.pas(et, q[:, None], k[:, None], v[:, None], g[:, None], b[:, None])[:, 0]
        ecarts.append(_ecart(o, E.pas_fp32(S, q, k, v, g, b)))
    assert max(ecarts) <= TOL_INT8, max(ecarts)
    assert sum(ecarts[-64:]) / 64 <= 2 * sum(ecarts[16:80]) / 64


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
