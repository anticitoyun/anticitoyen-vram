"""Échelle globale 2^k dans nvfp4_quant_act (poste7, revue/poste7-glm-pile-correctif-16-09.md
§ 1.4, cause : verdict-glm-saturation-16-09) et division AWQ fusionnée.

Sans échelle globale, un bloc de 16 dont amax/6 < 2⁻⁹ (plancher dénormal
E4M3) était écrit entièrement à zéro : 23-31 % des blocs de silu(g)·u à
l'entrée de down_proj sur GLM. Le noyau quantifie x·2^k et la GEMM compense
par gscale·2^-k. Contrats : noyau == référence Python de la même
arithmétique au bit (k, table) ; k = 0 casse sur des blocs petits, k = 8 non ;
la pile compensée rend le MÊME résultat qu'avant sur des blocs normaux
(2^k ne déplace que l'exposant E4M3) ; les compteurs disent 0 zéro / 0 saturé."""
import pytest
import torch

from tests.test_gemm_grouped_mma import quant_act_ref
from tests.test_moe_decode_mma_graphe import CUDA, INTER, TOP_K, _entree
from tests.test_moe_awq_pile import _bloc_awq, _boucle, _ulp_max

G, K, E = 48, 256, 4


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "nvfp4_quant_act"):
        pytest.skip("extension absente")
    return ext


def _x(dev, graine=3):
    g = torch.Generator().manual_seed(graine)
    x = torch.randn(G, K, generator=g)
    # blocs d'amplitudes étagées : normaux, petits (sous 2^-9·6 à k = 0),
    # grands (proches de la saturation à k = 8 : amax·2^8/6 ≤ 448 → amax ≤ 10,5)
    x.view(G, K // 16, 16)[::3] *= 2e-3
    x.view(G, K // 16, 16)[1::7] *= 3.0
    return x.to(dev, torch.bfloat16).contiguous()


def _table(dev, graine=4):
    g = torch.Generator().manual_seed(graine)
    awq = (0.5 + 1.5 * torch.rand(E, K, generator=g)).to(dev, torch.bfloat16).contiguous()
    es = torch.randint(0, E, (G,), generator=g).to(dev, torch.int32).contiguous()
    return awq, es


@CUDA
@pytest.mark.parametrize("log2k", [0, 4, 8])
@pytest.mark.parametrize("table", [False, True])
def test_noyau_egal_reference_au_bit(log2k, table):
    ext = _ext(); dev = torch.device("cuda:0")
    x = _x(dev)
    awq, es = _table(dev) if table else (None, None)
    cpt = torch.zeros(3, dtype=torch.int64, device=dev)
    xq, xsf = ext.nvfp4_quant_act(x, log2k, awq, es, cpt)
    xq_r, xsf_r = quant_act_ref(x, log2k, awq, es)
    assert torch.equal(xsf, xsf_r), f"échelles ≠ référence : {(xsf != xsf_r).sum().item()} blocs"
    assert torch.equal(xq, xq_r), f"codes ≠ référence : {(xq != xq_r).sum().item()} octets"
    # compteurs : blocs non nuls, mis à zéro (amax > 0 et échelle 0), saturés
    xf = x.float()
    if table:
        xf = (xf / awq[es.long()].float()).to(torch.bfloat16).float()
    amax = xf.view(G, K // 16, 16).abs().amax(-1) * 2 ** log2k
    n, z, sat = cpt.tolist()
    assert n == int((amax > 0).sum())
    assert z == int(((amax > 0) & (xsf == 0)).sum())
    assert sat == int((amax / 6 > 448).sum())


@CUDA
def test_l_echelle_globale_sauve_les_petits_blocs():
    """Un changement qui doit casser : à k = 0 les blocs petits sont mis à
    zéro (le défaut GLM) ; à k = 8 aucun, et la déquantification ·2^-8
    retrouve x au bruit E2M1 près."""
    ext = _ext(); dev = torch.device("cuda:0")
    g = torch.Generator().manual_seed(9)
    x = (torch.randn(G, K, generator=g) * 2e-3).to(dev, torch.bfloat16).contiguous()   # amax ≈ 6e-3 < 0,0117
    c0 = torch.zeros(3, dtype=torch.int64, device=dev); c8 = torch.zeros_like(c0)
    xq0, xsf0 = ext.nvfp4_quant_act(x, 0, None, None, c0)
    xq8, xsf8 = ext.nvfp4_quant_act(x, 8, None, None, c8)
    z0 = (xsf0 == 0).float().mean().item(); z8 = (xsf8 == 0).float().mean().item()
    assert z0 > 0.9, f"k=0 devait mettre à zéro presque tous les blocs : {z0:.2%}"
    assert z8 == 0.0 and c8[1].item() == 0 and c8[2].item() == 0, (z8, c8.tolist())
    assert c0[1].item() == int((xsf0 == 0).sum())
    from tests.test_gemm_grouped_mma import _dequant_nibbles
    deq = (_dequant_nibbles(xq8).float().view(G, K // 16, 16)
           * xsf8.view(torch.float8_e4m3fn).float().unsqueeze(-1)).view(G, K) * 2 ** -8
    err = ((deq - x.float()).norm() / x.float().norm()).item()
    assert err < 0.2, err                   # bruit E2M1 bloc 16 sur une gaussienne : ~0,1


@CUDA
def test_pile_compensee_identique_sur_blocs_normaux():
    """2^k ne déplace que l'exposant de l'E4M3 et gscale·2^-k est exact :
    sur des activations qui ne touchent ni le plancher ni le plafond, la
    pile rend le même bf16 au bit qu'avec k = 0 — le correctif ne change
    rien là où il n'y avait rien à corriger."""
    from acvram.engine import model as M
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks()
    bloc._stack_state = "oui"
    x, topw, topi = _entree(dev)
    y_k = bloc._forward_grouped_mma(x, topw, topi)
    kx, ka = M._QA_LOG2K_X, M._QA_LOG2K_ACT
    assert (kx, ka) == (4, 8), "défauts scellés (poste7 : k_x = 4, k_act = 8)"
    M._QA_LOG2K_X = M._QA_LOG2K_ACT = 0
    try:
        y_0 = bloc._forward_grouped_mma(x, topw, topi)
    finally:
        M._QA_LOG2K_X, M._QA_LOG2K_ACT = kx, ka
    torch.cuda.synchronize()
    assert y_k is not None and y_0 is not None
    diff = _ulp_max(y_k, y_0)
    # ≤ 1 ulp : seuls les blocs de silu(g)·u déjà sous le plancher à k = 0
    # (rares sur ces entrées) peuvent différer
    assert diff <= 1.0, f"{diff:.2f} ulp entre k=(4,8) et k=(0,0)"


@CUDA
def test_pile_k0_casse_sur_petites_activations():
    """Mêmes poids, x ÷ 16 : silu(g)·u tombe vers 1e-3 (amax/6 sous 2⁻⁹), à
    k = 0 la pile met ses blocs à zéro et s'écarte de la boucle W4A16 ; à
    k = (4, 8) aucun bloc à zéro et l'erreur reste celle du W4A4 — le test
    rend « faux » sur l'ancien noyau, et les compteurs disent pourquoi.
    (x ÷ 256 mettait aussi act·2^8 sous le plancher : la prémisse se vérifie
    par les compteurs, pas à l'œil — poste3, t-qa 17d7132.)"""
    from acvram.engine import model as M
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks()
    bloc._stack_state = "oui"
    x, topw, topi = _entree(dev)
    x = (x.float() * 2 ** -4).to(torch.bfloat16)
    ref = _boucle(bloc, x, topw, topi).float()
    kx, ka, compte = M._QA_LOG2K_X, M._QA_LOG2K_ACT, M._QA_COMPTE
    M._QA_COMPTE = True

    def passe(k_x, k_act):
        M._QA_LOG2K_X, M._QA_LOG2K_ACT = k_x, k_act
        M._QA_COMPTEURS.clear()
        y = bloc._forward_grouped_mma(x, topw, topi).float()
        torch.cuda.synchronize()
        n, z, sat = (int(v) for v in M._QA_COMPTEURS[x.device].tolist())
        return ((y - ref).norm() / ref.norm()).item(), n, z, sat
    try:
        e_k, n_k, z_k, sat_k = passe(4, 8)
        e_0, n_0, z_0, sat_0 = passe(0, 0)
    finally:
        M._QA_LOG2K_X, M._QA_LOG2K_ACT, M._QA_COMPTE = kx, ka, compte
        M._QA_COMPTEURS.clear()
    assert n_k == n_0 > 0
    # les compteurs cumulent les deux sites : x (1 quant, blocs normaux à
    # x ÷ 16) et act (G·I/16 blocs) ; la prémisse porte sur les blocs de act
    n_act = x.shape[0] * TOP_K * INTER // 16
    assert z_0 >= 0.8 * n_act, f"prémisse : k=0 devait mettre à zéro la plupart des blocs de act ({z_0}/{n_act})"
    assert z_k == 0 and sat_k == 0, f"k=(4,8) : {z_k} blocs à zéro, {sat_k} saturés sur {n_k}"
    assert e_0 > 3 * e_k, f"k=0 devait casser : err {e_0:.3f} contre {e_k:.3f} à k=(4,8)"
    assert e_k < 0.25, e_k
