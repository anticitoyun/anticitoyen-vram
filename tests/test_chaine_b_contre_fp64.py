"""Chantier 1 (Sage 20/09 13 h 45, après T4) — TEST D'INSTRUMENT, pas un correctif : l'erreur
relative de la chaîne B COMPLÈTE du MoE au décodage (quant act → GEMM groupée MMA FP4 → moe_act
→ requant → reduce, ``MoEBlock._forward_grouped_mma`` sans fusion) contre une référence float64
NON quantifiée sur l'activation (x fp64, activation fp64 ; seuls les poids sont ceux du modèle,
déquantifiés nvfp4 → fp64), PAR LIGNE, sur les formes servies :
  * Coder-30B-A3B : hidden 2 048, inter 768, top-8, lot b=12 (16 experts construits, voir FORMES) ;
  * GLM-4.7-Flash : hidden 2 048, inter 1 536, top-4, t = 5 (t ≥ 5 : lot MMA).
Le chiffre mesuré à l'écriture (20/09, extension du dépôt) est PUBLIÉ dans ``MESURE_20_09`` et le
scellé vaut ≤ 1,5 × ce chiffre (médiane des erreurs relatives par ligne) ; il entre dans
MECANISMES (« mma-a4 coûte ≈ X % d'erreur relative par couche, absorbée par le routage »).
Faute construite : l'échelle globale des poids de down_proj × f. Mesuré à l'écriture : × 1,02 ne bouge
la médiane que de 0,05 pt (résolution entre deux graines 0,3-0,4 pt) — UNE ERREUR DE 16 % NE VOIT PAS 2 %
(la faute de Sage est sous la résolution de ce juge, et on le dit) ; le test cherche le plus petit facteur
de [1,02 ; 1,05 ; 1,10 ; 1,20] qui déplace la médiane de plus de 3 × la résolution et exige qu'il existe.
"""
import pytest
pytestmark = pytest.mark.pile_naturelle
import torch
from acvram.quant.nvfp4 import quantize_nvfp4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")
# experts=16 (pas 128/64) : la forme qui compte pour l'erreur par ligne est celle des GEMM (hidden × inter, k experts par
# jeton, t jetons) ; 384 QuantLinear de 2 048 × 768 coûtent > 10 min à construire (quantize_nvfp4 à l'écriture, 20/09).
FORMES = {"coder-b12": dict(hidden=2048, inter=768, experts=16, k=8, t=12),
          "glm-t5": dict(hidden=2048, inter=1536, experts=16, k=4, t=5)}
# Mesuré à l'écriture (20/09, main 776b59ad, sm_120) : médiane de l'erreur relative L2 par ligne.
# À remplir par la première passe (voir verdict) ; None = le test publie sans juger.
MESURE_20_09 = {"coder-b12": 0.1599, "glm-t5": 0.1648}   # 20/09 14 h 02 : 15,99 % (max 17,64) et 16,48 % (max 17,24)
FACTEUR_SCELLE = 1.5


def _bloc(dev, f, graine=1):
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.kernels import get_extension
    from acvram.quant.q3n import quantize_q3n
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("noyau MMA FP4 indisponible")

    def lin(sortie, entree, g_):
        g = torch.Generator().manual_seed(g_)
        w = (torch.randn(sortie, entree, generator=g) * 0.05).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w.to(dev)), out_features=sortie, in_features=entree).to_device(dev)
    experts = [MLP(lin(f["inter"], f["hidden"], graine * 1000 + 10 * e + 1), lin(f["inter"], f["hidden"], graine * 1000 + 10 * e + 2),
                   lin(f["hidden"], f["inter"], graine * 1000 + 10 * e + 3)) for e in range(f["experts"])]
    g = torch.Generator().manual_seed(999 + graine)
    routeur = QuantLinear(quantize_q3n(torch.randn(f["experts"], f["hidden"], generator=g) * 0.02),
                          out_features=f["experts"], in_features=f["hidden"]).to_device(dev)
    bloc = MoEBlock(routeur, experts, f["k"]).to(dev)
    assert bloc._try_build_stacks(), bloc._raison_repli
    bloc._stack_state = "oui"
    return bloc


def _entree(dev, f, graine=7):
    g = torch.Generator().manual_seed(graine)
    x = (torch.randn(f["t"], f["hidden"], generator=g) * 0.5).to(dev, torch.bfloat16)
    topi = torch.stack([torch.randperm(f["experts"], generator=g)[:f["k"]] for _ in range(f["t"])]).to(dev)
    topw = torch.softmax(torch.randn(f["t"], f["k"], generator=g), -1).to(dev)
    return x, topw, topi


def _b(bloc, x, topw, topi):
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    a = MOE._MOE_DECODE_FUSED
    try:
        MOE._MOE_DECODE_FUSED = False
        return bloc._forward_grouped_mma(x, topw, topi)
    finally:
        MOE._MOE_DECODE_FUSED = a


def _ref_fp64_non_quantifiee(bloc, x, topw, topi, faute_down=1.0):
    """x fp64 tel quel (PAS nvfp4_quant_act), activation fp64 (PAS requantifiée) ; poids = ceux du
    bloc, déquantifiés nvfp4 → fp64 ; ``faute_down`` multiplie l'échelle globale de down_proj."""
    from tests.test_gemm_grouped_mma import _w64
    pg, pu, pd = (bloc._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
    W = {n: _w64(p[1], p[2]).reshape(p[1].shape[0], p[1].shape[1], -1).double() for n, p in (("g", pg), ("u", pu), ("d", pd))}
    t, k = topi.shape
    xa = x.double()
    y = torch.zeros(t, pd[1].shape[1], dtype=torch.float64, device=x.device)
    for i in range(t):
        for j in range(k):
            e = int(topi[i, j])
            g = (xa[i] @ W["g"][e].T) * pg[3][e].double()
            u = (xa[i] @ W["u"][e].T) * pu[3][e].double()
            act = g / (1 + torch.exp(-g)) * u
            y[i] += (act @ W["d"][e].T) * pd[3][e].double() * faute_down * float(topw[i, j])
    return y


def _erreur_par_ligne(y, ref):
    return ((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30))


@CUDA
@pytest.mark.parametrize("forme", list(FORMES))
def test_chaine_b_contre_fp64(forme):
    dev = torch.device("cuda:0"); f = FORMES[forme]
    bloc = _bloc(dev, f)
    x, topw, topi = _entree(dev, f)
    yb = _b(bloc, x, topw, topi)
    ref = _ref_fp64_non_quantifiee(bloc, x, topw, topi)
    err = _erreur_par_ligne(yb, ref)
    med, mx = float(err.median()), float(err.max())
    # résolution : la même chaîne sur une autre entrée (graine 8) — l'écart des médianes est le bruit du juge
    x2, topw2, topi2 = _entree(dev, f, 8)
    err2 = _erreur_par_ligne(_b(bloc, x2, topw2, topi2), _ref_fp64_non_quantifiee(bloc, x2, topw2, topi2))
    resolution = abs(float(err2.median()) - med)
    # faute construite : échelle globale de down_proj × f dans la RÉFÉRENCE (= la chaîne B sortirait ×1/f de trop)
    fautes = {}
    for fac in (1.02, 1.05, 1.10, 1.20):
        fautes[fac] = float(_erreur_par_ligne(yb, _ref_fp64_non_quantifiee(bloc, x, topw, topi, faute_down=fac)).median()) - med
    vue = next((fac for fac in sorted(fautes) if abs(fautes[fac]) > 3 * resolution), None)
    print(f"\n[chaine B vs fp64 non quantifié] {forme} t={f['t']} : erreur relative L2 par ligne médiane {100 * med:.2f} % "
          f"(max {100 * mx:.2f} %, graine 8 : {100 * float(err2.median()):.2f} %, résolution {100 * resolution:.2f} pt) ; "
          f"faute down : " + ", ".join(f"×{fac:.2f} {100 * d:+.2f} pt" for fac, d in fautes.items()) + f" ; plus petite vue (> 3 × résolution) : {vue}")
    assert torch.isfinite(err).all()
    if MESURE_20_09[forme] is not None:
        assert med <= FACTEUR_SCELLE * MESURE_20_09[forme], \
            f"{forme} : médiane {100 * med:.2f} % > {FACTEUR_SCELLE} × {100 * MESURE_20_09[forme]:.2f} % (mesuré à l'écriture)"
    # le juge doit voir UNE faute construite : le plus petit facteur vu est publié (20/09 : ×1,02 invisible, cf. docstring)
    assert vue is not None, f"le juge ne voit aucune échelle ≤ ×1,20 : {fautes} contre une résolution de {100 * resolution:.2f} pt"
