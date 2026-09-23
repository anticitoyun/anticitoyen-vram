"""Rotation de Hadamard des experts (Sage, revue/sage-hadamard-16-09.md) :
poids tournés W·H par le convertisseur (`hadamard_block` du manifeste, 512
sur GLM), entrée tournée x·H par bloc à l'exécution — dans nvfp4_quant_act
(FWHT fp32 en mémoire partagée) sur le chemin MMA, en torch
(fwht_activations) sur GEMV / direct / _grouped_mm, et dans
ChannelScaler.apply pour la boucle par expert. Contrats : noyau == référence
Python au bit (FWHT puis E2M1) ; la pile GEMV (W4A16) rend la boucle à ≤ 1 ulp
avec des experts tournés ; les chemins W4A4 restent au bruit de quantification
(≤ 1,25× le cas non tourné) ; x·H·(W·H)ᵀ = x·Wᵀ (orthogonalité, fp32)."""
import pytest

pytestmark = pytest.mark.pile_naturelle   # lit _stacks[nom][1], rendu (None) sous Marlin, défaut servi (T4 20/09)
import torch

from tests.test_gemm_grouped_mma import quant_act_ref
from tests.test_moe_awq_pile import _boucle, _ulp_max
from tests.test_moe_decode_mma_graphe import CUDA, N_EXPERTS, TOP_K, CACHE, INTER, _entree
from tests.test_quant_act_echelle import _ext, _x, _table, G, K


@CUDA
@pytest.mark.parametrize("bloc", [16, 64, 256])
@pytest.mark.parametrize("table", [False, True])
def test_noyau_fwht_egal_reference_au_bit(bloc, table):
    ext = _ext(); dev = torch.device("cuda:0")
    x = _x(dev)
    awq, es = _table(dev) if table else (None, None)
    xq, xsf, gr = ext.nvfp4_quant_act(x, awq, es, None, bloc)
    xq_r, xsf_r, gr_r = quant_act_ref(x, awq, es, hadamard=bloc)
    assert torch.equal(gr, gr_r) and torch.equal(xsf, xsf_r), "échelles ≠ référence après FWHT"
    assert torch.equal(xq, xq_r), f"codes ≠ référence : {(xq != xq_r).sum().item()} octets"
    # sans rotation les codes diffèrent : le drapeau agit
    xq0, _, _ = ext.nvfp4_quant_act(x, awq, es, None, 0)
    assert not torch.equal(xq0, xq)


@CUDA
def test_fwht_activations_orthogonale_et_egale_au_noyau():
    """fwht_activations (torch, boucle et chemins bf16 de la pile) : même
    résultat que le noyau (via la référence) et orthogonale (x·H·(W·H)ᵀ = x·Wᵀ)."""
    from acvram.quant.calibrate import fwht_activations
    from tests.test_gemm_grouped_mma import fwht_ref
    dev = torch.device("cuda:0")
    x = _x(dev)
    a = fwht_activations(x, 256)
    assert torch.equal(a.float(), fwht_ref(x.float(), 256))
    xf = torch.randn(8, 1024, device=dev); w = torch.randn(64, 1024, device=dev)
    xh = fwht_activations(xf, 512); wh = fwht_activations(w, 512)
    assert torch.allclose(xh @ wh.T, xf @ w.T, rtol=1e-4, atol=1e-3)


def _bloc_tourne(dev, hd_x=512, hd_d=256):
    """Le bloc MoE du test AWQ, sans échelle, experts tournés : W·H bloc-diagonale
    avant quantification, scaler hadamard_block sur chaque projection."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.quant.q3n import quantize_q3n
    from acvram.quant.nvfp4 import quantize_nvfp4
    from acvram.quant.calibrate import ChannelScaler, fwht_activations
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("noyau MMA FP4 indisponible")

    def lin(sortie, entree, graine, bloc):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(sortie, entree, generator=g) * 0.05).to(dev)
        w[:, :4] *= 20.0                                   # canaux aberrants : ce que la rotation étale
        wh = fwht_activations(w, bloc)                      # W·H (H symétrique)
        sc = ChannelScaler(scale=None, hadamard_block=bloc)
        return QuantLinear(quantize_nvfp4(wh.to(torch.bfloat16)), out_features=sortie, in_features=entree,
                           scaler=sc).to_device(dev)
    experts = [MLP(lin(INTER, CACHE, 10 * e + 1, hd_x), lin(INTER, CACHE, 10 * e + 2, hd_x),
                   lin(CACHE, INTER, 10 * e + 3, hd_d)) for e in range(N_EXPERTS)]
    gen = torch.Generator().manual_seed(5)
    routeur = QuantLinear(quantize_q3n(torch.randn(N_EXPERTS, CACHE, generator=gen) * 0.02),
                          out_features=N_EXPERTS, in_features=CACHE).to_device(dev)
    return MoEBlock(routeur, experts, TOP_K).to(dev)


@CUDA
@pytest.mark.parametrize("chemin", ["gemv", "mma", "prefill_mma", "prefill_direct"])
def test_pile_tournee_egale_boucle(chemin, monkeypatch):
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    dev = torch.device("cuda:0")
    bloc = _bloc_tourne(dev)
    assert bloc._try_build_stacks(), bloc._raison_repli
    assert bloc._stacks_awq["hadamard"] == {"gate_proj": 512, "up_proj": 512, "down_proj": 256}
    bloc._stack_state = "oui"
    x, topw, topi = _entree(dev)
    ref = _boucle(bloc, x, topw, topi)                     # ChannelScaler.apply tourne l'entrée
    if chemin.startswith("prefill"):
        monkeypatch.setattr(MOE, "_MOE_MMA", chemin == "prefill_mma")
        monkeypatch.setattr(MOE, "_MOE_GEMM_MAX", 1e9)
    if chemin == "mma":
        y = bloc._forward_grouped_mma(x, topw, topi)
    elif chemin == "gemv":
        y = bloc._forward_grouped(x, topw, topi)
    else:
        y = bloc._forward_prefill_grouped(x, topw, topi)
    assert y is not None
    ecart = _ulp_max(y, ref)
    if chemin in ("gemv", "prefill_direct"):
        assert ecart <= 1.0, f"{chemin} tourné : {ecart:.2f} ulp de la boucle"
    else:
        # W4A4 : bruit de quantification ; témoin de faute : la pile SANS
        # rotation de l'entrée (poids tournés, x non tourné) doit être loin
        assert ecart < 64, f"{chemin} tourné : {ecart:.1f} ulp"
        sauve = bloc._stacks_awq
        bloc._stacks_awq = {**sauve, "hadamard": {}}
        y_bad = (bloc._forward_grouped_mma if chemin == "mma" else bloc._forward_prefill_grouped)(x, topw, topi)
        bloc._stacks_awq = sauve
        assert _ulp_max(y_bad, ref) > 4 * max(ecart, 1.0), "sans rotation de l'entrée la pile devrait diverger"


@CUDA
def test_fwht_activations_capturable_dans_un_graphe():
    """Manon (verdict-glm-hadamard-conversion) : sur le converti tourné, 0 godet
    capturé — la normalisation créait un tenseur depuis l'hôte à chaque appel
    (copie CPU→carte interdite en capture). fwht_activations doit se capturer
    et rendre, au rejeu, exactement le résultat eager."""
    from acvram.quant.calibrate import fwht_activations
    dev = torch.device("cuda:0")
    x = torch.randn(12, 2048, device=dev).to(torch.bfloat16)
    attendu = fwht_activations(x, 512).clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwht_activations(x, 512)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fwht_activations(x, 512)
    y.zero_(); g.replay(); torch.cuda.synchronize()
    assert torch.equal(y, attendu)
