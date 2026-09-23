"""Échelle globale PAR LIGNE dans nvfp4_quant_act (Sage, revue/sage-glm-pile-correctif-16-09.md
§ 7, cause : verdict-glm-saturation-16-09, k fixe réfuté par verdict-reppl-alpha-commun-16-09)
et division AWQ fusionnée.

Sans échelle globale, un bloc de 16 dont amax/6 < 2⁻⁹ (plancher dénormal
E4M3) était écrit entièrement à zéro : 23-31 % des blocs de silu(g)·u à
l'entrée de down_proj sur GLM ; un 2^k fixe saturait les queues d'une fenêtre
réelle. Le noyau pose g_r = amax_r/2688 par ligne, s_blk = (amax_blk/amax_r)×448 : le bloc maximal tombe
sur 448 (saturation impossible), un bloc n'est flushé que sous ~2,2e-6 × amax_r.
Contrats : noyau == référence Python de la même arithmétique au bit (table
comprise) ; 0 saturé quelle que soit l'amplitude ; invariance d'échelle par
ligne ; les compteurs le prouvent ; la pile reste près de la boucle W4A16 sur
des activations petites (là où l'ancien noyau rendait ~100 % d'erreur)."""
import pytest

pytestmark = pytest.mark.pile_naturelle   # lit _stacks[nom][1], rendu (None) sous Marlin, défaut servi (T4 20/09)
import torch

from tests.test_gemm_grouped_mma import quant_act_ref, dequant_act_ref
from tests.test_moe_decode_mma_graphe import CUDA, INTER, TOP_K, _entree
from tests.test_moe_awq_pile import _bloc_awq, _boucle

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
    # blocs d'amplitudes étagées dans une même ligne : normaux, petits
    # (flushés par l'ancien noyau), grands (saturés par un 2^k fixe)
    x.view(G, K // 16, 16)[::3] *= 2e-3
    x.view(G, K // 16, 16)[1::7] *= 3.0
    x[5, :16] *= 3e3                          # queue : amax·2^8/6 ≫ 448
    x[9] = 0                                  # ligne nulle : g_r = 0, tout à zéro
    x[11, 16:] = 0                            # un seul bloc non nul
    return x.to(dev, torch.bfloat16).contiguous()


def _table(dev, graine=4):
    g = torch.Generator().manual_seed(graine)
    awq = (0.5 + 1.5 * torch.rand(E, K, generator=g)).to(dev, torch.bfloat16).contiguous()
    es = torch.randint(0, E, (G,), generator=g).to(dev, torch.int32).contiguous()
    return awq, es


@CUDA
@pytest.mark.parametrize("table", [False, True])
def test_noyau_egal_reference_au_bit(table):
    ext = _ext(); dev = torch.device("cuda:0")
    x = _x(dev)
    awq, es = _table(dev) if table else (None, None)
    cpt = torch.zeros(3, dtype=torch.int64, device=dev)
    xq, xsf, gr = ext.nvfp4_quant_act(x, awq, es, cpt)
    xq_r, xsf_r, gr_r = quant_act_ref(x, awq, es)
    assert torch.equal(gr, gr_r), f"échelles de ligne ≠ référence : {(gr != gr_r).sum().item()} lignes"
    assert torch.equal(xsf, xsf_r), f"échelles ≠ référence : {(xsf != xsf_r).sum().item()} blocs"
    assert torch.equal(xq, xq_r), f"codes ≠ référence : {(xq != xq_r).sum().item()} octets"
    # compteurs : blocs non nuls (ligne non nulle), flushés (amax > 0, échelle 0), saturés (jamais)
    xf = x.float()
    if table:
        xf = (xf / awq[es.long()].float()).to(torch.bfloat16).float()
    amax = xf.view(G, K // 16, 16).abs().amax(-1)
    vivant = (amax > 0) & (gr > 0).unsqueeze(-1)
    n, z, sat = cpt.tolist()
    assert n == int(vivant.sum())
    assert z == int((vivant & (xsf == 0)).sum())
    assert sat == 0
    assert gr[9].item() == 0.0 and int(xsf[9].sum()) == 0 and int(xq[9].sum()) == 0
    assert xsf[11, 0].item() == 0x7E, "le bloc maximal d'une ligne porte l'échelle 448 (E4M3 0x7E)"


@CUDA
def test_jamais_sature_et_petits_blocs_gardes():
    """Un changement qui doit casser sur les deux anciens noyaux : sans échelle
    globale, les blocs à 2e-3 étaient flushés ; avec 2^k fixe, une queue à
    3e3 saturait. Ici : 0 saturé, 0 flushé tant que amax_blk ≥ 4e-6 × amax_r,
    et la déquantification retrouve x au bruit E2M1 près."""
    ext = _ext(); dev = torch.device("cuda:0")
    g = torch.Generator().manual_seed(9)
    x = torch.randn(G, K, generator=g) * 2e-3                     # amax ≈ 6e-3 : flushé par l'ancien noyau
    x[7, :16] *= 5e5                                              # queue à ~3e3 : saturée par 2^8
    x = x.to(dev, torch.bfloat16).contiguous()
    cpt = torch.zeros(3, dtype=torch.int64, device=dev)
    xq, xsf, gr = ext.nvfp4_quant_act(x, None, None, cpt)
    n, z, sat = cpt.tolist()
    assert sat == 0
    amax = x.float().view(G, K // 16, 16).abs().amax(-1)
    amax_r = amax.amax(-1, keepdim=True)
    # flush ⇔ (amax_blk/amax_r)·448 < 2^-10 (sous l'E4M3) ⇔ amax_blk < 2,2e-6 × amax_r
    petits_gardes = (amax >= 4e-6 * amax_r) & (amax > 0)
    assert int(((xsf == 0) & petits_gardes).sum()) == 0, "un bloc ≥ 4e-6 × amax de ligne a été flushé"
    assert z == int(((xsf == 0) & (amax > 0)).sum())
    deq = dequant_act_ref(xq, xsf, gr)
    lignes = [i for i in range(G) if i != 7]
    err = ((deq[lignes] - x[lignes].double()).norm() / x[lignes].double().norm()).item()
    assert err < 0.2, err                   # bruit E2M1 bloc 16 sur une gaussienne : ~0,1
    err7 = ((deq[7, :16] - x[7, :16].double()).norm() / x[7, :16].double().norm()).item()
    assert err7 < 0.2, f"la queue est écrasée : {err7:.3f}"


@CUDA
def test_invariance_d_echelle_par_ligne():
    """x et x·2^10 : mêmes codes, mêmes échelles de bloc, g_r × 1024 exactement."""
    ext = _ext(); dev = torch.device("cuda:0")
    x = _x(dev)
    xq, xsf, gr = ext.nvfp4_quant_act(x)
    xq2, xsf2, gr2 = ext.nvfp4_quant_act((x.float() * 1024.0).to(torch.bfloat16).contiguous())
    assert torch.equal(xq, xq2) and torch.equal(xsf, xsf2)
    assert torch.equal(gr2, gr * 1024.0)


@CUDA
def test_pile_petites_activations_pres_de_la_boucle():
    """Mêmes poids, x ÷ 16 : silu(g)·u vers 1e-3, là où l'ancien noyau mettait
    la majorité des blocs de act à zéro (err ≈ 1 contre la boucle W4A16). La
    pile corrigée reste dans la marge W4A4, sans saturation, avec un flush
    résiduel négligeable — les compteurs le disent."""
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks()
    bloc._stack_state = "oui"
    x, topw, topi = _entree(dev)
    compte = MOE._QA_COMPTE
    MOE._QA_COMPTE = True
    try:
        res = {}
        for nom, f in (("normal", 1.0), ("petit", 2 ** -4)):
            xf = (x.float() * f).to(torch.bfloat16)
            ref = _boucle(bloc, xf, topw, topi).float()
            MOE._QA_COMPTEURS.clear()
            y = bloc._forward_grouped_mma(xf, topw, topi).float()
            torch.cuda.synchronize()
            n, z, sat = (int(v) for v in MOE._QA_COMPTEURS[x.device].tolist())
            res[nom] = (((y - ref).norm() / ref.norm()).item(), n, z, sat)
    finally:
        MOE._QA_COMPTE = compte
        MOE._QA_COMPTEURS.clear()
    n_act = x.shape[0] * TOP_K * INTER // 16
    for nom, (e, n, z, sat) in res.items():
        assert sat == 0, f"{nom} : {sat} blocs saturés"
        assert z <= 0.01 * n, f"{nom} : {z} blocs flushés sur {n}"
        assert e < 0.25, f"{nom} : err {e:.3f} contre la boucle"
    assert n_act > 0 and res["petit"][0] < 2 * res["normal"][0] + 0.05, res
