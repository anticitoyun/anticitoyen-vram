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


# --- C15 niveau 3 : logits du routeur DANS le noyau (route_logits_fusee) ----
# Sous l'interpréteur les entrées sont fp16 (bf16 n'y est pas porté : dot et
# arrondi rendent n'importe quoi, mesuré le 20/09) et l'arrondi bf16 des logits
# n'est pas rejoué ; sur carte, bf16 et arrondi comme en service. Le bras
# bf16 + ARRONDI_BF16 n'a donc de preuve que sur carte (REGLES § 7).

def _dtype_essai():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float16


@pytest.mark.parametrize("sigmoide,biais,renorm,scale", [(False, False, True, 1.0), (True, True, True, 2.5),
                                                          (True, False, False, 1.0)])
@pytest.mark.parametrize("E,k,H", [(128, 8, 256), (60, 6, 96), (256, 8, 64)])
def test_c15_logits_fusee_egale_linear_puis_route_fusee(sigmoide, biais, renorm, scale, E, k, H):
    """Témoin : F.linear (produits exacts, somme fp32) puis `route_fusee`.
    Mêmes experts, mêmes eid et compteur au bit, poids à 2⁻¹⁶ près (ordre de
    la somme fp32 des logits, puis exp) ; godet 20 (2 programmes) avec 4
    fantômes masqués par `valid`."""
    rp = _rp()
    dt = _dtype_essai()
    torch.manual_seed(E + k + H)
    T = 20
    x = (torch.randn(T, H) * 0.5).to(DEV, dt)
    w = (torch.randn(E, H) * 0.05).to(DEV, dt)
    bias = (torch.randn(E) * 0.2).to(DEV) if biais else None
    valid = torch.ones(T, dtype=torch.bool, device=DEV); valid[16:] = False
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    tw, ti, eid = rp.route_logits_fusee(x, w, bias, k, sigmoide, renorm, scale, valid, u, arrondi_bf16=False)
    lg = torch.nn.functional.linear(x.float(), w.float())
    u2 = torch.zeros(E, dtype=torch.int64, device=DEV)
    rw, ri, reid = rp.route_fusee(lg, bias, k, sigmoide, renorm, scale, valid, u2)
    assert torch.equal(ti, ri), (ti[:3], ri[:3])
    assert torch.equal(eid, reid) and (eid.view(T, k)[16:] == -1).all()
    assert torch.equal(u, u2)
    assert (tw - rw).abs().max() < 2 ** -16 * max(1.0, scale), float((tw - rw).abs().max())


def test_c15_logits_fusee_un_poids_deplace_change_la_selection():
    """Le bras qui doit casser : la ligne 0 du routeur remplacée par la ligne
    de l'expert le plus choisi — logits égaux, l'égalité va à l'indice le
    plus bas (moe_route) : la sélection suit les poids lus (pas un cache, pas
    un index de programme), l'expert 0 passe devant lui partout où il était
    choisi, et le compteur ne le perd pas."""
    rp = _rp()
    dt = _dtype_essai()
    torch.manual_seed(7)
    T, H, E, k = 12, 128, 64, 4
    x = (torch.randn(T, H) * 0.5).to(DEV, dt)
    w = (torch.randn(E, H) * 0.05).to(DEV, dt)
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    _, ti, _ = rp.route_logits_fusee(x, w, None, k, False, True, 1.0, None, u, arrondi_bf16=False)
    favori = int(u[1:].argmax()) + 1
    w2 = w.clone(); w2[0] = w[favori]
    u2 = torch.zeros(E, dtype=torch.int64, device=DEV)
    _, ti2, _ = rp.route_logits_fusee(x, w2, None, k, False, True, 1.0, None, u2, arrondi_bf16=False)
    assert not torch.equal(ti, ti2) and int(u2[0]) >= int(u[0])
    choisi = (ti == favori).any(1)                         # sur ces lignes, 0 est classé avant favori
    rang0 = (ti2 == 0).int().argmax(1); rangf = (ti2 == favori).int().argmax(1)
    assert ((rang0 < rangf) | ~(ti2 == favori).any(1))[choisi].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="arrondi bf16 : carte seulement")
def test_c15_logits_fusee_arrondi_bf16_comme_cublas():
    """Sur carte : ARRONDI_BF16 rejoue la sortie bf16 de F.linear — mêmes
    experts (hors égalité à l'ulp : aucune sur cette graine) et poids top-k
    à 2⁻¹² près de cuBLAS puis route_fusee."""
    rp = _rp()
    torch.manual_seed(3)
    T, H, E, k = 12, 2048, 128, 8
    x = (torch.randn(T, H) * 0.5).to(DEV, torch.bfloat16)
    w = (torch.randn(E, H) * 0.02).to(DEV, torch.bfloat16)
    u = torch.zeros(E, dtype=torch.int64, device=DEV)
    tw, ti, _ = rp.route_logits_fusee(x, w, None, k, False, False, 1.0, None, u, arrondi_bf16=True)
    lg = torch.nn.functional.linear(x, w)
    u2 = torch.zeros(E, dtype=torch.int64, device=DEV)
    rw, ri, _ = rp.route_fusee(lg, None, k, False, False, 1.0, None, u2)
    assert torch.equal(ti, ri)
    assert (tw - rw).abs().max() < 2 ** -12, float((tw - rw).abs().max())
