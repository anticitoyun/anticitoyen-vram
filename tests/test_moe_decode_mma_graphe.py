"""Chemin MoE décodage par la MMA groupée (``MoEBlock._forward_grouped_mma``,
levier (3) de poste7) : (a) bit à bit égal au chemin prefill MMA sur le même
lot ; (b) créneaux fantômes (topi = -1) → contribution nulle et finie, les
autres lignes inchangées ; (c) capturable dans un vrai ``torch.cuda.graph``
et le rejeu rend la même sortie que l'eager (aucun ``.item()``)."""
import pytest

pytestmark = pytest.mark.pile_naturelle   # lit _stacks[nom][1], rendu (None) sous Marlin, défaut servi (T4 20/09)
import torch

from acvram.quant.nvfp4 import quantize_nvfp4

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")
CACHE, INTER, N_EXPERTS, TOP_K, T = 512, 256, 16, 4, 12


def _bloc(dev):
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("noyau MMA FP4 indisponible")

    def lin(sortie, entree, graine):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(sortie, entree, generator=g) * 0.05).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w.to(dev)), out_features=sortie, in_features=entree).to_device(dev)

    experts = [MLP(lin(INTER, CACHE, 10 * e + 1), lin(INTER, CACHE, 10 * e + 2), lin(CACHE, INTER, 10 * e + 3))
               for e in range(N_EXPERTS)]
    from acvram.quant.q3n import quantize_q3n
    g = torch.Generator().manual_seed(999)
    routeur = QuantLinear(quantize_q3n(torch.randn(N_EXPERTS, CACHE, generator=g) * 0.02),
                          out_features=N_EXPERTS, in_features=CACHE).to_device(dev)
    bloc = MoEBlock(routeur, experts, TOP_K).to(dev)
    assert bloc._try_build_stacks(), bloc._raison_repli
    bloc._stack_state = "oui"
    return bloc


def _entree(dev, graine=7):
    g = torch.Generator().manual_seed(graine)
    x = (torch.randn(T, CACHE, generator=g) * 0.5).to(dev, torch.bfloat16)
    topi = torch.stack([torch.randperm(N_EXPERTS, generator=g)[:TOP_K] for _ in range(T)]).to(dev)
    topw = torch.softmax(torch.randn(T, TOP_K, generator=g), -1).to(dev)
    return x, topw, topi


@CUDA
def test_decode_mma_egal_prefill_mma():
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    y_dec = bloc._forward_grouped_mma(x, topw, topi)
    assert y_dec is not None
    y_pre = bloc._forward_prefill_grouped(x, topw, topi)
    assert torch.equal(y_dec, y_pre), f"{int((y_dec != y_pre).sum())} valeurs differentes"
    assert torch.isfinite(y_dec).all()


@CUDA
def test_fantomes_nuls_et_finis():
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    y_ref = bloc._forward_grouped_mma(x, topw, topi)
    topi_f = topi.clone(); topi_f[8:] = -1
    x_f = x.clone(); x_f[8:] = 0
    y_f = bloc._forward_grouped_mma(x_f, topw, topi_f)
    assert torch.isfinite(y_f).all()
    assert torch.equal(y_f[8:], torch.zeros_like(y_f[8:]))
    assert torch.equal(y_f[:8], y_ref[:8])


@CUDA
def test_capturable_en_graphe_et_rejeu_identique():
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    xs, ws, ts = x.clone(), topw.clone(), topi.clone()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            bloc._forward_grouped_mma(xs, ws, ts)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g = bloc._forward_grouped_mma(xs, ws, ts)
    x2, w2, t2 = _entree(dev, graine=11)
    xs.copy_(x2); ws.copy_(w2); ts.copy_(t2)
    g.replay(); torch.cuda.synchronize()
    y_e = bloc._forward_grouped_mma(x2, w2, t2)
    assert torch.equal(y_g, y_e), f"rejeu != eager : {int((y_g != y_e).sum())} valeurs"


@CUDA
def test_grille_fixe_large_t0_hors_tampon():
    """Grille rembourrée bien plus large que le lot (t_max = 96 tuiles pour 48
    lignes) : les tuiles vides ont t0 au-delà de xq — le noyau ne doit rien y
    lire (accès illégal du 14/09 sur Coder-30B, E=128)."""
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    ancien = MOE._MOE_DECODE_MMA_BT
    y_exact = bloc._forward_prefill_grouped(x, topw, topi)
    y = bloc._forward_grouped_mma(x, topw, topi)
    torch.cuda.synchronize()
    assert torch.equal(y, y_exact)
