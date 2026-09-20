"""MoE fusionné au décodage (port b12x, `nvfp4_moe_fused`) :
(a) DÉTERMINISME : deux appels identiques rendent la même sortie au bit
    (split-K sériel, ordre fixe des tranches — contrôle des jumelles, REGLES §4) ;
(b) contre le chemin B (3 GEMM + moe_act + quant + reduce) : même intermédiaire
    (FC1+quant bit-identiques par construction), FC2 sommé dans un autre ordre
    → égal à la référence float64 à la tolérance bf16, et ≥ 95 % des sorties
    bit-identiques à B ;
(c) fantômes : contribution nulle et finie ; (d) tn 64 et 128 concordent au bit
    entre eux ? non (ordre des tranches différent) — chacun contre float64."""
import pytest

pytestmark = pytest.mark.pile_naturelle   # lit _stacks[nom][1], rendu (None) sous Marlin, défaut servi (T4 20/09)
import torch

from tests.test_moe_decode_mma_graphe import _bloc, _entree, CUDA, N_EXPERTS, TOP_K, T, CACHE, INTER


def _fused(bloc, x, topw, topi, tn):
    from acvram.engine import model as M
    a, b = M._MOE_DECODE_FUSED, M._MOE_FUSED_TN
    try:
        M._MOE_DECODE_FUSED, M._MOE_FUSED_TN = True, tn
        return bloc._forward_grouped_mma(x, topw, topi)
    finally:
        M._MOE_DECODE_FUSED, M._MOE_FUSED_TN = a, b


def _b(bloc, x, topw, topi):
    from acvram.engine import model as M
    a = M._MOE_DECODE_FUSED
    try:
        M._MOE_DECODE_FUSED = False
        return bloc._forward_grouped_mma(x, topw, topi)
    finally:
        M._MOE_DECODE_FUSED = a


def _quant_act_sans_echelle_globale(v: torch.Tensor) -> torch.Tensor:
    """Aller-retour de l'activation telle que le noyau FUSIONNÉ la requantifie
    en shared : amax/6 -> E4M3 (satfinite 448), valeurs / échelle -> E2M1 au
    plus proche (égalités vers le code pair), SANS échelle globale de ligne —
    l'arithmétique d'avant poste7-glm-pile-correctif § 7, gardée sur ce chemin
    témoin (un CTA ne voit qu'une tranche de I). float64 [N]."""
    from tests.test_gemm_grouped_mma import _MID, _E2M1
    vb = v.float().reshape(-1, 16)
    s = (vb.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    ok = s > 0
    q = torch.where(ok.unsqueeze(-1), vb / s.clamp(min=1e-30).unsqueeze(-1), torch.zeros_like(vb))
    a = q.abs().clamp(max=6.0)
    mid = _MID.to(v.device)
    iu = torch.searchsorted(mid, a.reshape(-1).contiguous(), right=True).view_as(a)
    il = torch.searchsorted(mid, a.reshape(-1).contiguous(), right=False).view_as(a)
    code = torch.where(iu != il, torch.where(il % 2 == 0, il, iu), iu)
    val = _E2M1.to(v.device)[code] * torch.sign(q)
    return (val * s.unsqueeze(-1)).double().reshape(-1)


def _ref64(bloc, x, topw, topi):
    """Référence float64 depuis les MÊMES tenseurs quantifiés que le noyau
    fusionné : x par nvfp4_quant_act (échelle de ligne g_r, consommée par
    l'épilogue FC1), l'activation requantifiée sans échelle globale comme
    dans le noyau."""
    from tests.test_gemm_grouped_mma import dequant_act_ref, _w64
    from acvram.kernels import get_extension
    ext = get_extension()
    pg, pu, pd = (bloc._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
    W = {n: _w64(p[1], p[2]).reshape(p[1].shape[0], p[1].shape[1], -1) for n, p in (("g", pg), ("u", pu), ("d", pd))}
    t, k = topi.shape
    y = torch.zeros(t, pd[1].shape[1], dtype=torch.float64, device=x.device)
    xq, xsf, gr = ext.nvfp4_quant_act(x.to(torch.bfloat16).contiguous())
    xa = dequant_act_ref(xq, xsf, gr)
    for i in range(t):
        for j in range(k):
            e = int(topi[i, j])
            if e < 0:
                continue
            g = (xa[i] @ W["g"][e].T) * pg[3][e].double()
            u = (xa[i] @ W["u"][e].T) * pu[3][e].double()
            g, u = g.to(torch.bfloat16).double(), u.to(torch.bfloat16).double()
            act = (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16)
            aa = _quant_act_sans_echelle_globale(act)
            y[i] += (aa @ W["d"][e].T) * pd[3][e].double() * float(topw[i, j])
    return y


@CUDA
@pytest.mark.parametrize("tn", [64, 128])
def test_fused_deterministe_et_juste(tn):
    from acvram.kernels import get_extension
    if not hasattr(get_extension(), "nvfp4_moe_fused"):
        pytest.skip("extension sans nvfp4_moe_fused")
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    y1 = _fused(bloc, x, topw, topi, tn)
    y2 = _fused(bloc, x, topw, topi, tn)
    assert torch.isfinite(y1).all()
    assert torch.equal(y1, y2), "deux appels identiques different : le split-K n'est pas seriel"
    ref = _ref64(bloc, x, topw, topi)
    yb = _b(bloc, x, topw, topi)
    tol = ref.abs() * 2 ** -6 + 1e-2 * ref.abs().max()
    hors = int(((y1.double() - ref).abs() > tol).sum())
    assert hors == 0, f"{hors} sorties hors tolerance vs float64 (max {(y1.double() - ref).abs().max().item():.3e})"
    # Le chemin B quantifie l'activation avec une échelle globale par ligne
    # (nvfp4_quant_act, poste7-glm-pile-correctif § 7) ; le noyau fusionné la
    # requantifie en shared sans (un CTA ne voit qu'une tranche de I) : les
    # codes E2M1 diffèrent, l'identité au bit avec B n'est plus un contrat —
    # seule la tolérance float64 l'est.
    # REGLES § 7 (T4 20/09 : 2 634 sorties « hors tolérance vs chemin B », max 4,5e-2, le fusionné
    # étant DANS la tolérance float64) : deux approximations ne se jugent pas l'une contre l'autre ;
    # le chemin B se juge, lui aussi, contre float64 — un rouge ici nomme B, pas le fusionné.
    hors_b = int(((yb.double() - ref).abs() > tol).sum())
    assert hors_b == 0, (f"chemin B : {hors_b} sorties hors tolerance vs float64 (max {(yb.double() - ref).abs().max().item():.3e}) ; "
                         f"fusionné − B max {(y1.double() - yb.double()).abs().max().item():.3e}")


@CUDA
def test_fused_fantomes():
    from acvram.kernels import get_extension
    if not hasattr(get_extension(), "nvfp4_moe_fused"):
        pytest.skip("extension sans nvfp4_moe_fused")
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    y_ref = _fused(bloc, x, topw, topi, 64)
    topi_f = topi.clone(); topi_f[8:] = -1; x_f = x.clone(); x_f[8:] = 0
    y_f = _fused(bloc, x_f, topw, topi_f, 64)
    assert torch.isfinite(y_f).all()
    assert torch.equal(y_f[8:], torch.zeros_like(y_f[8:]))
    assert torch.equal(y_f[:8], y_ref[:8])
