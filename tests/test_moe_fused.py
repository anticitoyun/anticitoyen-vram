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
    from acvram.engine import moe as MOE
    a, b = MOE._MOE_DECODE_FUSED, MOE._MOE_FUSED_TN
    try:
        MOE._MOE_DECODE_FUSED, MOE._MOE_FUSED_TN = True, tn
        return bloc._forward_grouped_mma(x, topw, topi)
    finally:
        MOE._MOE_DECODE_FUSED, MOE._MOE_FUSED_TN = a, b


def _b(bloc, x, topw, topi):
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    a = MOE._MOE_DECODE_FUSED
    try:
        MOE._MOE_DECODE_FUSED = False
        return bloc._forward_grouped_mma(x, topw, topi)
    finally:
        MOE._MOE_DECODE_FUSED = a


def _quant_act_sans_echelle_globale(v: torch.Tensor) -> torch.Tensor:
    """Aller-retour de l'activation telle que le noyau FUSIONNÉ la requantifie
    en shared : amax/6 -> E4M3 (satfinite 448), valeurs / échelle -> E2M1 au
    plus proche (égalités vers le code pair), SANS échelle globale de ligne —
    l'arithmétique d'avant sage-glm-pile-correctif § 7, gardée sur ce chemin
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


def _ref(bloc, x, topw, topi, dtype=torch.float64, inverse=False, faute=None):
    """Référence depuis les MÊMES tenseurs quantifiés que le noyau fusionné : x par
    nvfp4_quant_act (échelle de ligne g_r, consommée par l'épilogue FC1), l'activation
    requantifiée sans échelle globale comme dans le noyau. ``dtype`` float64 = la
    référence ; float32 avec ``inverse`` (experts accumulés du dernier au premier) =
    un second ordre de somme légitime, le TÉMOIN du juge (§ 7) ; ``faute`` = (expert,
    facteur) : la sortie de cet expert multipliée — la faute construite."""
    from tests.test_gemm_grouped_mma import dequant_act_ref, _w64
    from acvram.kernels import get_extension
    ext = get_extension()
    pg, pu, pd = (bloc._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
    W = {n: _w64(p[1], p[2]).reshape(p[1].shape[0], p[1].shape[1], -1).to(dtype) for n, p in (("g", pg), ("u", pu), ("d", pd))}
    t, k = topi.shape
    y = torch.zeros(t, pd[1].shape[1], dtype=dtype, device=x.device)
    xq, xsf, gr = ext.nvfp4_quant_act(x.to(torch.bfloat16).contiguous())
    xa = dequant_act_ref(xq, xsf, gr).to(dtype)
    for i in range(t):
        for j in (range(k - 1, -1, -1) if inverse else range(k)):
            e = int(topi[i, j])
            if e < 0:
                continue
            g = (xa[i] @ W["g"][e].T) * pg[3][e].to(dtype)
            u = (xa[i] @ W["u"][e].T) * pu[3][e].to(dtype)
            g, u = g.to(torch.bfloat16).to(dtype), u.to(torch.bfloat16).to(dtype)
            act = (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16)
            aa = _quant_act_sans_echelle_globale(act).to(dtype)
            contrib = (aa @ W["d"][e].T) * pd[3][e].to(dtype) * float(topw[i, j])
            if faute is not None and e == faute[0]:
                contrib = contrib * faute[1]
            y[i] += contrib
    return y


def _ref64(bloc, x, topw, topi):
    return _ref(bloc, x, topw, topi)


def _ulp_bf16(r):
    r = r.abs().double()
    return torch.where(r > 0, 2.0 ** (torch.floor(torch.log2(r.clamp_min(1e-300))) - 7), torch.full_like(r, 2.0 ** -133))


def juge_fuse_contre_b(y_fuse, y_b, ref64, temoin):
    """§ 7 (Sage 13 h 12) : d = |y − ref64| en ulp bf16 de l'AMPLITUDE de la ligne (max |ref| de la ligne ;
    en ulp de chaque élément, les sorties proches de zéro rendaient 6 000 ulp sans rien dire), max par ligne ;
    tenu si (1) d(fusé) ≤ d(B) + 2·témoin sur chaque ligne ET (2) médiane(fusé) ≤ médiane(B) ET
    (3) max(fusé) ≤ max(B) + 1 ET (4, resserré : la clause que la faute construite casse) d(fusé) ≤ 1 + 2·témoin
    — 1 ulp de l'amplitude = l'arrondi bf16 de la sortie, mesuré 0,83 le 20/09 ; un expert ×1,02 rend 2,97.
    Rend (tenu, texte)."""
    u = _ulp_bf16(ref64.abs().amax(-1, keepdim=True))
    d_f = ((y_fuse.double() - ref64).abs() / u).amax(-1)
    d_b = ((y_b.double() - ref64).abs() / u).amax(-1)
    lignes = int((d_f > d_b + 2 * temoin).sum())
    med_ok = d_f.median().item() <= d_b.median().item()
    max_ok = d_f.max().item() <= d_b.max().item() + 1.0
    serre = int((d_f > 1.0 + 2 * temoin).sum())
    texte = (f"d(fusé) médiane {d_f.median().item():.2f} max {d_f.max().item():.2f} ; d(B) médiane {d_b.median().item():.2f} "
             f"max {d_b.max().item():.2f} ulp bf16 de l'amplitude ; témoin {temoin:.3f} ; lignes fusé > d(B) + 2·témoin : "
             f"{lignes}/{d_f.numel()} ; lignes fusé > 1 + 2·témoin : {serre}/{d_f.numel()}")
    return lignes == 0 and med_ok and max_ok and serre == 0, texte


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
    # Le chemin B quantifie l'activation avec une échelle globale par ligne (nvfp4_quant_act,
    # sage-glm-pile-correctif § 7) ; le fusionné la requantifie en shared sans : les codes E2M1
    # diffèrent, l'identité au bit avec B n'est pas un contrat. § 7 (Sage 13 h 12, après T4 20/09 :
    # 2 634 « hors tolérance vs B ») : fusionné et B CHACUN contre float64 par ligne ; témoin = deux
    # ordres de somme légitimes de la référence en fp32 (experts du premier au dernier / inverse).
    r32a, r32b = _ref(bloc, x, topw, topi, torch.float32), _ref(bloc, x, topw, topi, torch.float32, inverse=True)
    temoin = ((r32a.double() - r32b.double()).abs() / _ulp_bf16(ref.abs().amax(-1, keepdim=True))).max().item()
    tenu, texte = juge_fuse_contre_b(y1, yb, ref, temoin)
    print(f"\n[moe_fused tn={tn}] {texte}")
    assert tenu, texte
    # faute construite : la sortie d'UN expert ×1,02 (dans l'arithmétique de référence, fp32) doit casser —
    # mesuré 20/09 : 2,97 ulp de l'amplitude (×1,05 : 7,4) contre 0,83 pour le fusionné sain ; B, qui quantifie
    # l'activation avec une échelle globale de ligne, est à 13-23 ulp de CETTE référence (pas la sienne) : les
    # clauses (1)-(3) ne voient pas 2 %, la clause (4) oui — résolution du juge : détecte ≥ 2 %
    e0 = int(topi[0, 0])
    y_faute = _ref(bloc, x, topw, topi, torch.float32, faute=(e0, 1.02))
    casse, texte_f = juge_fuse_contre_b(y_faute, yb, ref, temoin)
    assert not casse, f"le juge ne voit pas un expert ×1,02 : {texte_f}"


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
