"""Échelle AWQ PAR EXPERT dans la pile (poste7, revue/poste7-glm-awq-pile-15-09.md) :
le loader gardait la boucle par expert dès qu'un expert avait une échelle ;
il construit maintenant une table [E, K] par projection et l'applique aux
lignes rassemblées (route+pack / moe_act pour le chemin MMA, x[tok]/s[e] pour
le GEMV b<9). Contrat : pile == boucle par expert (QuantLinear.forward :
x / s en bf16) à ≤ 1 ulp bf16 ; route+pack avec AWQ == torch au bit."""
import pytest
import torch

from acvram.quant.nvfp4 import quantize_nvfp4
from acvram.quant.calibrate import ChannelScaler
from tests.test_moe_decode_mma_graphe import CUDA, N_EXPERTS, TOP_K, T, CACHE, INTER, _entree


def _bloc_awq(dev, gate_up_egales=True):
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.quant.q3n import quantize_q3n
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("noyau MMA FP4 indisponible")
    gen = torch.Generator().manual_seed(5)

    def lin(sortie, entree, graine, echelle):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(sortie, entree, generator=g) * 0.05).to(torch.bfloat16)
        sc = ChannelScaler(scale=echelle.to(torch.float16), hadamard_block=0) if echelle is not None else None
        return QuantLinear(quantize_nvfp4(w.to(dev)), out_features=sortie, in_features=entree, scaler=sc).to_device(dev)

    experts = []
    for e in range(N_EXPERTS):
        s_in = (0.5 + torch.rand(CACHE, generator=gen)) if e % 3 else None     # un expert sur trois sans échelle
        s_up = s_in if gate_up_egales else (0.5 + torch.rand(CACHE, generator=gen))
        s_dn = 0.5 + torch.rand(INTER, generator=gen)
        experts.append(MLP(lin(INTER, CACHE, 10 * e + 1, s_in), lin(INTER, CACHE, 10 * e + 2, s_up),
                           lin(CACHE, INTER, 10 * e + 3, s_dn)))
    routeur = QuantLinear(quantize_q3n(torch.randn(N_EXPERTS, CACHE, generator=gen) * 0.02),
                          out_features=N_EXPERTS, in_features=CACHE).to_device(dev)
    return MoEBlock(routeur, experts, TOP_K).to(dev)


def _boucle(bloc, x, topw, topi):
    """La référence : QuantLinear.forward par expert (x / s puis GEMV), sommée par jeton."""
    y = torch.zeros(x.shape[0], CACHE, dtype=torch.float32, device=x.device)
    for i in range(x.shape[0]):
        for j in range(TOP_K):
            e = int(topi[i, j])
            if e < 0:
                continue
            m = bloc.experts[e]
            xi = x[i:i + 1]
            g = m.gate_proj(xi); u = m.up_proj(xi)
            a = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
            y[i] += m.down_proj(a).float()[0] * float(topw[i, j])
    return y.to(torch.bfloat16)


def _ulp_max(a, b):
    """écart max en ulp bf16 du plus grand |b| (l'ulp d'un élément proche de
    zéro rendrait des millions pour une différence d'arrondi sans portée)."""
    ulp = b.abs().float().max() * 2 ** -7
    return ((a.float() - b.float()).abs().max() / ulp).item()


def _sans_echelle(bloc):
    """La boucle avec les scalers retirés : référence du témoin sans table."""
    for m in bloc.experts:
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(m, n).scaler = None


@CUDA
def test_pile_acceptee_et_tables():
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks(), bloc._raison_repli
    t = bloc._stacks_awq
    assert t["gate_proj"] is not None and t["down_proj"] is not None
    assert t["gate_proj"].shape == (N_EXPERTS, bloc._stacks["gate_proj"][4])
    assert torch.equal(t["gate_proj"][0], torch.ones_like(t["gate_proj"][0]))     # expert 0 sans échelle


@CUDA
def test_gate_up_differents_refuses():
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev, gate_up_egales=False)
    assert not bloc._try_build_stacks()
    assert "gate" in bloc._raison_repli


@CUDA
@pytest.mark.parametrize("chemin", ["mma", "gemv"])
def test_pile_egale_boucle_a_un_ulp(chemin):
    from acvram.engine import model as M
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks()
    bloc._stack_state = "oui"
    x, topw, topi = _entree(dev)
    ref = _boucle(bloc, x, topw, topi)
    if chemin == "mma":
        y = bloc._forward_grouped_mma(x, topw, topi)
    else:
        y = bloc._forward_grouped(x, topw, topi)
    assert y is not None
    ecart = _ulp_max(y, ref)
    if chemin == "gemv":
        # W4A16 des deux côtés : la pile doit rendre la boucle à l'ulp près
        assert ecart <= 1.0, f"GEMV avec AWQ par expert : {ecart:.2f} ulp de la boucle"
    # la pile MMA quantifie l'activation (W4A4) là où la boucle est W4A16 :
    # l'écart attendu est celui de la quantification, présent SANS échelle.
    # Contrôle de l'ÉCHELLE : le même écart, normalisé par |ref|, sans table
    # contre la boucle sans scalers ne doit pas être plus petit que celui
    # avec table contre la boucle avec scalers (rapport ≤ 1,25).
    sauve = bloc._stacks_awq
    bloc._stacks_awq = {}
    y_bad = bloc._forward_grouped_mma(x, topw, topi) if chemin == "mma" else bloc._forward_grouped(x, topw, topi)
    ecart_bad = _ulp_max(y_bad, ref)
    _sans_echelle(bloc)
    ref0 = _boucle(bloc, x, topw, topi)
    y0 = bloc._forward_grouped_mma(x, topw, topi) if chemin == "mma" else bloc._forward_grouped(x, topw, topi)
    bloc._stacks_awq = sauve
    ecart0 = _ulp_max(y0, ref0)
    # témoin de faute : sans table, la pile est loin de la boucle échelonnée
    assert ecart_bad > 2 * max(ecart, 1.0), f"témoin sans table trop proche ({ecart_bad:.1f} vs {ecart:.1f} ulp)"
    assert ecart <= 1.25 * max(ecart0, 1.0), \
        f"{chemin} avec AWQ : {ecart:.2f} ulp de la boucle, contre {ecart0:.2f} sans échelle"


@CUDA
def test_route_pack_awq_egal_torch():
    from acvram.kernels import get_extension
    ext = get_extension()
    dev = torch.device("cuda:0")
    bloc = _bloc_awq(dev)
    assert bloc._try_build_stacks()
    x, topw, topi = _entree(dev)
    topi = topi.clone(); topi[10:] = -1
    E, bt = N_EXPERTS, 16; t_max = -(-(T * TOP_K) // bt) + E
    pg = bloc._stacks["gate_proj"]; awq = bloc._stacks_awq["gate_proj"]
    xs, ordre, *_ , es = ext.moe_route_pack(topi.contiguous(), topw.contiguous(), x.contiguous(), E, bt, t_max, pg[4], awq)
    flat_e = torch.where(topi < 0, torch.zeros_like(topi), topi).reshape(-1)
    o = torch.argsort(flat_e, stable=True)
    ref = (x.to(torch.bfloat16)[torch.arange(T, device=dev).repeat_interleave(TOP_K)[o]] / awq[flat_e[o]])
    assert torch.equal(xs, ref)
    assert torch.equal(es, flat_e[o].to(torch.int32))
