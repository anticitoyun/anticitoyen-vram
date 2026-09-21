"""Condition (i) de Sage (revue/sage-moe-mma-decodage-14-09.md) pour le MoE
en MMA groupée au décodage : un jeton doit recevoir, du chemin « décodage »
(lot de 12 jetons, tuiles de 16 par expert presque vides) exactement ce
qu'il reçoit du chemin « prefill » (les mêmes 12 jetons noyés dans un lot
de 200, tuiles pleines) — bit à bit, pour les trois projections et pour
la sortie du bloc entier. La qualité du W4A4 est déjà payée (+0,919 %,
A4) ; ce test garantit qu'elle ne se paie pas deux fois.
"""
import pytest
import torch

from tests.test_gemm_grouped_mma import _ext, _pile, _tables, _tuiles, _x

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _routage(t, E, k, graine):
    g = torch.Generator(device="cpu").manual_seed(graine)
    return torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(t)]).cuda()


def _gemm_par_routage(ext, pile, xq, xsf, gr, topi, bt):
    """Sortie [t*k, M] dans l'ordre (jeton, expert) d'origine, comme le fait
    _forward_prefill_grouped : tri stable par expert, GEMM groupée, retour."""
    qw, bs, gs = pile
    E = qw.shape[0]
    flat_e = topi.reshape(-1).to(torch.int64)
    ordre = torch.argsort(flat_e, stable=True)
    cnt = torch.bincount(flat_e, minlength=E)
    te, t0, tn = _tuiles(cnt, bt)
    tq, tb = _tables(qw, bs)
    tok = torch.arange(topi.shape[0], device="cuda").repeat_interleave(topi.shape[1])[ordre]
    M, K = qw.shape[1], 2 * qw.shape[2]                 # qw : [E, M, K/2] (deux E2M1 par octet)
    y = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq[tok].contiguous(), xsf[tok].contiguous(),
                                   te, t0, tn, M, K, bt, 4, grow=gr[tok].contiguous())
    inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device="cuda")
    return y[inv]


@pytest.mark.parametrize("bt", [16, 64])
def test_decodage_mma_egal_prefill_mma(bt):
    ext = _ext()
    E, k, K, M = 32, 8, 1536, 768
    pile = _pile(E, M, K, 0.05, 41)
    # 12 jetons « décodage » + 188 autres : le prefill est le lot des 200.
    x_all = _x(200, K, 7)
    topi_all = _routage(200, E, k, 3)
    xq, xsf, gr = ext.nvfp4_quant_act(x_all)
    y_prefill = _gemm_par_routage(ext, pile, xq, xsf, gr, topi_all, bt)
    xq12, xsf12, gr12 = ext.nvfp4_quant_act(x_all[:12].contiguous())
    assert torch.equal(xq12, xq[:12]) and torch.equal(xsf12, xsf[:12]) and torch.equal(gr12, gr[:12])   # la quantification est par ligne
    y_decode = _gemm_par_routage(ext, pile, xq12, xsf12, gr12, topi_all[:12], bt)
    # les deux côtés contre la référence float64 (sinon deux sorties fausses seraient « égales »)
    from tests.test_gemm_grouped_mma import dequant_act_ref, _w64
    qw, bs, gs = pile
    xa = dequant_act_ref(xq12, xsf12, gr12)
    w = _w64(qw, bs).reshape(E, M, K)
    ref = torch.stack([(xa[t] @ w[e].T) * gs[e].double() for t in range(12) for e in topi_all[t].tolist()])
    tol = ref.abs() * 2 ** -6 + 1e-2
    assert int(((y_decode.double() - ref).abs() > tol).sum()) == 0
    assert torch.equal(y_decode, y_prefill[:12 * k]), \
        f"{int((y_decode != y_prefill[:12 * k]).sum())} valeurs differentes entre decodage et prefill"


def test_changement_qui_casse():
    """Le contrôle peut rendre faux : une autre pile donne une autre sortie."""
    ext = _ext()
    E, k, K, M = 8, 2, 512, 256
    x = _x(12, K, 9); topi = _routage(12, E, k, 5)
    xq, xsf, gr = ext.nvfp4_quant_act(x)
    a = _gemm_par_routage(ext, _pile(E, M, K, 0.05, 1), xq, xsf, gr, topi, 16)
    b = _gemm_par_routage(ext, _pile(E, M, K, 0.05, 2), xq, xsf, gr, topi, 16)
    assert not torch.equal(a, b)
