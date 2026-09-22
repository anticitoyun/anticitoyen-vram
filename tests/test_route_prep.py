"""Juge de la fusion (1) du poste F (kernels/route_prep.py) : `eid`, le
compteur d'usage des experts et les index de jetons doivent être ceux du
chemin torch (`masked_fill(~valid, -1)`, `_compter_routage`, `arange`,
`repeat_interleave`) AU BIT, sur les godets 1/2/8/16 avec fantômes, avec et
sans `valid`, compteur cumulé sur plusieurs pas ; et un bras qui doit
casser : un fantôme non masqué change le compteur. Sans carte :
``TRITON_INTERPRET=1`` (conftest)."""
import importlib
import os

import pytest
import torch


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rp():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    rp = importlib.import_module("acvram.kernels.route_prep")
    if not rp.disponible():
        pytest.skip("Triton indisponible")
    return rp


def _torch(topi, valid, usage):
    """Le chemin torch d'aujourd'hui (model.py : forward + _compter_routage + _forward_grouped)."""
    if valid is not None:
        topi = topi.masked_fill(~valid.unsqueeze(-1), -1)
    idx = topi.reshape(-1).to(torch.int64)
    usage.scatter_add_(0, idx.clamp(min=0), (idx >= 0).to(torch.int64))
    t, k = topi.shape
    eid = topi.reshape(-1).to(torch.int32)
    tok = torch.arange(t, dtype=torch.int32, device=topi.device).repeat_interleave(k)
    seq = torch.arange(t * k, dtype=torch.int32, device=topi.device)
    return eid, tok, seq


@pytest.mark.parametrize("godet", [1, 2, 8, 16])
@pytest.mark.parametrize("avec_valid", [True, False])
def test_eid_compteur_et_index_au_bit(godet, avec_valid):
    rp = _rp()
    torch.manual_seed(godet)
    E, k = 128, 8
    usage_t = torch.zeros(E, dtype=torch.int64, device=DEV)
    usage_r = torch.zeros(E, dtype=torch.int64, device=DEV)
    for pas in range(3):                                   # compteur cumulé sur trois pas
        topi = torch.randint(0, E, (godet, k), dtype=torch.int32, device=DEV)
        valid = None
        if avec_valid:
            valid = torch.ones(godet, dtype=torch.bool, device=DEV)
            valid[max(1, godet - godet // 4):] = False      # les dernières lignes sont des fantômes
        eid_t, tok_t, seq_t = _torch(topi.clone(), valid, usage_t)
        eid_r = rp.route_prep(topi, valid, usage_r)
        tok_r, seq_r = rp.index_jetons(godet, k, topi.device)
        assert torch.equal(eid_r, eid_t) and eid_r.dtype == torch.int32
        assert torch.equal(tok_r, tok_t) and torch.equal(seq_r, seq_t)
        assert torch.equal(usage_r, usage_t), (usage_r - usage_t).nonzero()
    assert rp.index_jetons(godet, k, topi.device)[0] is tok_r, "les index sont réservés une fois par godet"


def test_un_fantome_non_masque_casse_le_compteur():
    """Le bras qui doit différer : sans `valid`, les fantômes comptent — le
    juge voit la différence, donc il voit aussi un masque oublié."""
    rp = _rp()
    topi = torch.randint(0, 16, (8, 4), dtype=torch.int32, device=DEV)
    valid = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool, device=DEV)
    avec, sans = torch.zeros(16, dtype=torch.int64, device=DEV), torch.zeros(16, dtype=torch.int64, device=DEV)
    e1 = rp.route_prep(topi, valid, avec)
    e2 = rp.route_prep(topi, None, sans)
    assert int(avec.sum()) == 24 and int(sans.sum()) == 32
    assert (e1 == -1).sum() == 8 and (e2 == -1).sum() == 0


def _route_torch(lg, bias, k, sigmoide, renorm, scale):
    """La branche torch de `MoEBlock._route` (celle qui remplace moe_route sans extension)."""
    if sigmoide:
        probs = torch.sigmoid(lg)
        sel = probs if bias is None else probs + bias
        _, topi = torch.topk(sel, k, dim=-1)
        topw = probs.gather(-1, topi)
    else:
        probs = torch.softmax(lg, dim=-1)
        topw, topi = torch.topk(probs, k, dim=-1)
    if renorm:
        topw = topw / topw.sum(dim=-1, keepdim=True)
    return topw * scale, topi


@pytest.mark.parametrize("sigmoide,biais,renorm,scale", [(False, False, True, 1.0), (True, True, True, 2.5),
                                                          (True, False, False, 1.0), (False, False, False, 1.0)])
@pytest.mark.parametrize("E,k", [(128, 8), (256, 8), (60, 6), (1024, 32)])
def test_f2_route_fusee_egale_moe_route(sigmoide, biais, renorm, scale, E, k):
    """F2 : mêmes experts (égalités vers l'indice le plus bas, comme
    moe_route_kernel), poids à 2⁻²⁰ près (exp fp32 : libdevice contre __expf),
    eid et compteur au bit, sur des godets avec fantômes."""
    rp = _rp()
    torch.manual_seed(E + k)
    T = 16
    lg = (torch.randn(T, E) * 3).to(DEV)
    lg[3, :4] = lg[3, 5]                                   # égalités fabriquées
    bias = (torch.randn(E) * 0.2).to(DEV) if biais else None
    valid = torch.ones(T, dtype=torch.bool, device=DEV); valid[12:] = False
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    tw, ti, eid = rp.route_fusee(lg, bias, k, sigmoide, renorm, scale, valid, u)
    rw, ri = _route_torch(lg, bias, k, sigmoide, renorm, scale)
    assert torch.equal(ti.long(), ri), (ti[3], ri[3])
    assert (tw - rw).abs().max() < 2 ** -20 * max(1.0, scale), float((tw - rw).abs().max())
    assert torch.equal(eid.view(T, k)[:12].long(), ri[:12]) and (eid.view(T, k)[12:] == -1).all()
    attendu = torch.zeros(E, dtype=torch.int64, device=DEV).scatter_add_(0, ri[:12].reshape(-1), torch.ones(12 * k, dtype=torch.int64, device=DEV))
    assert torch.equal(u, attendu)


def test_f2_un_biais_change_la_selection_mais_pas_les_poids():
    """Le bras qui doit différer (moe_route : sélection sur probs + biais,
    poids = probs sans biais)."""
    rp = _rp()
    torch.manual_seed(1)
    lg = torch.randn(4, 64).to(DEV)
    bias = torch.zeros(64, device=DEV); bias[7] = 10.0     # l'expert 7 est forcé
    u = torch.zeros(64, dtype=torch.int64, device=DEV)
    tw, ti, _ = rp.route_fusee(lg, bias, 4, True, False, 1.0, None, u)
    assert (ti == 7).any(1).all()
    probs = torch.sigmoid(lg)
    assert torch.allclose(tw, probs.gather(-1, ti.long()), atol=2 ** -20)


# --- C15-3c : routeur compact = mêmes logits que le témoin + sélection à 1 warp ----
# (verdict-c15-niveau3-coder-19-09 addendum 03 h 17 : les versions Triton du GEMM
# des logits — 0ca85673, 3e62a9b5 — donnaient |Δtopw| 10⁻³ et des experts ≠ A :
# le témoin sort ses logits en bf16 (`_router_logits`, model.py : dtype_voulu = x.dtype
# pour softmax sans biais), toute différence d'ordre de somme fp32 devient un saut de
# 2⁻⁸ à une frontière d'arrondi ; seul le MÊME appel cuBLAS donne les mêmes logits.)

def _dtype_essai():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float16


def _poids_fp64(lg, k, sigmoide, renorm, scale, topi):
    """Les poids de `_route_torch` en fp64 pour la sélection `topi` DÉJÀ jugée identique
    (les égalités se tranchent en fp32 ; la référence ne re-sélectionne pas)."""
    probs = torch.sigmoid(lg.double()) if sigmoide else torch.softmax(lg.double(), dim=-1)
    topw = probs.gather(-1, topi)
    if renorm:
        topw = topw / topw.sum(dim=-1, keepdim=True)
    return topw * scale


def _ulp32_de(x, ref64):
    """Distance de x (fp32) à la référence fp64, en ulp fp32 de la référence, par élément."""
    r = ref64.to(torch.float64)
    ulp = torch.where(r != 0, 2.0 ** (torch.floor(torch.log2(r.abs().clamp_min(1e-300))) - 23),
                      torch.full_like(r, 2.0 ** -149))
    return (x.double() - r).abs() / ulp


def _ulp32_max(a, b):
    a32, b32 = a.float(), b.float()
    ulp = torch.where(a32 != 0, 2.0 ** (torch.floor(torch.log2(a32.abs().clamp_min(1e-30))) - 23),
                      torch.full_like(a32, 2.0 ** -149))
    return float(((a32 - b32).abs() / ulp).max())


@pytest.mark.parametrize("w_fp32", [False, True])
@pytest.mark.parametrize("sigmoide,biais,renorm,scale", [(False, False, True, 1.0), (True, True, True, 2.5)])
def test_c15_3c_routage_identique_au_temoin_sur_512_jetons(w_fp32, sigmoide, biais, renorm, scale):
    """Témoin A = `_router_logits` (F.linear au dtype du poids : bf16 — fp16 à
    sec — ou fp32) puis `route_fusee` (4 warps) ; B = `route_logits_fusee` (le
    même F.linear, sélection à 1 warp). 512 jetons de bruit, E 128, H 2 048 :
    experts (topi) et eid IDENTIQUES, compteur au bit, |Δtopw| ≤ 4 ulp fp32
    (deux dispositions de la même somme fp32 du softmax ; à sec, l'interpréteur
    ne connaît pas les warps : 0 ulp)."""
    rp = _rp()
    dt = torch.float32 if w_fp32 else _dtype_essai()
    torch.manual_seed(512 + int(w_fp32) + int(sigmoide))
    T, H, E, k = 512, 2048, 128, 8
    x = (torch.randn(T, H) * 0.5).to(DEV, _dtype_essai())
    w = (torch.randn(E, H) * 0.02).to(DEV, dt)
    bias = (torch.randn(E) * 0.2).to(DEV) if biais else None
    valid = torch.ones(T, dtype=torch.bool, device=DEV); valid[500:] = False
    lg = torch.nn.functional.linear(x.to(dt), w)                   # _router_logits, au bit
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    rw, ri, reid = rp.route_fusee(lg, bias, k, sigmoide, renorm, scale, valid, u)
    u2 = torch.zeros(E, dtype=torch.int64, device=DEV)
    tw, ti, eid = rp.route_logits_fusee(x, w, bias, k, sigmoide, renorm, scale, valid, u2)
    assert torch.equal(ti, ri) and torch.equal(eid, reid) and torch.equal(u, u2)
    # REGLES § 7 (sage-t4-tri-69-20-09) : deux approximations fp32 ne se jugent pas
    # l'une contre l'autre (T4 20/09 : 5 ulp entre elles, aucune fautive) ; chacune
    # contre la référence fp64 de la même sélection, et l'écart des deux ≤ 6 ulp.
    ref64 = _poids_fp64(lg, k, sigmoide, renorm, scale, ri.long())
    d_a, d_b = _ulp32_de(rw, ref64), _ulp32_de(tw, ref64)
    assert float((d_b - d_a).max()) <= 6.0 and float((d_a - d_b).max()) <= 6.0, \
        f"témoin {float(d_a.max()):.2f} ulp, fusé {float(d_b.max()):.2f} ulp contre fp64 ; écart max {float((d_b - d_a).abs().max()):.2f} > 6"


def test_c15_3c_un_poids_deplace_change_la_selection():
    """Le bras qui doit casser : la ligne 0 du routeur remplacée par la ligne
    de l'expert le plus choisi — logits égaux, l'égalité va à l'indice le
    plus bas : la sélection suit les poids lus, l'expert 0 passe devant lui."""
    rp = _rp()
    dt = _dtype_essai()
    torch.manual_seed(7)
    T, H, E, k = 12, 128, 64, 4
    x = (torch.randn(T, H) * 0.5).to(DEV, dt)
    w = (torch.randn(E, H) * 0.05).to(DEV, dt)
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    _, ti, _ = rp.route_logits_fusee(x, w, None, k, False, True, 1.0, None, u)
    favori = int(u[1:].argmax()) + 1
    w2 = w.clone(); w2[0] = w[favori]
    u2 = torch.zeros(E, dtype=torch.int64, device=DEV)
    _, ti2, _ = rp.route_logits_fusee(x, w2, None, k, False, True, 1.0, None, u2)
    assert not torch.equal(ti, ti2) and int(u2[0]) >= int(u[0])
    choisi = (ti == favori).any(1)
    rang0 = (ti2 == 0).int().argmax(1); rangf = (ti2 == favori).int().argmax(1)
    assert ((rang0 < rangf) | ~(ti2 == favori).any(1))[choisi].all()


def test_c15_3c_le_temoin_rend_visible_tout_ecart_de_somme_a_2_moins_8():
    """Pourquoi un GEMM Triton des logits ne peut pas rendre le routage du
    témoin : le témoin arrondit ses logits en bf16 (`_router_logits`) ; un
    autre ordre de somme fp32 fait basculer, à une frontière d'arrondi, UN
    logit d'un ulp bf16 (2⁻⁸ relatif) — et ce seul ulp déplace le poids top-k
    de ~10⁻³ (le |Δtopw| 1,06 × 10⁻³ du 19/09) ; à égalité proche il change
    l'expert. Arithmétique torch seule (softmax fp32 du témoin)."""
    torch.manual_seed(3)
    lg = (torch.randn(512, 128) * 1.5).to(torch.bfloat16)
    probs = torch.softmax(lg.float(), -1)
    tw, ti = probs.topk(8, -1)
    lg2 = lg.clone()
    i0 = ti[:, 0]                                             # le meilleur expert de chaque jeton : + 1 ulp bf16
    v = lg2[torch.arange(512), i0].float()
    ulp = 2.0 ** (torch.floor(torch.log2(v.abs().clamp_min(1e-30))) - 7)
    lg2[torch.arange(512), i0] = (v + ulp).to(torch.bfloat16)
    assert (lg2 != lg).sum() == 512
    tw2, _ = torch.softmax(lg2.float(), -1).topk(8, -1)
    assert float((tw2 - tw).abs().max()) > 5e-4 and float((tw2[:, 0] - tw[:, 0]).abs().median()) > 1e-4
    # ... et sur 65 536 logits, une somme fp32 dans un autre ordre en fait basculer au moins un :
    x = (torch.randn(512, 2048) * 0.5).to(torch.bfloat16); w = (torch.randn(128, 2048) * 0.02).to(torch.bfloat16)
    a = torch.nn.functional.linear(x.float(), w.float()).to(torch.bfloat16)                  # une somme fp32
    b = torch.nn.functional.linear(x.double(), w.double()).to(torch.bfloat16)                # la somme exacte
    assert int((a != b).sum()) > 0
