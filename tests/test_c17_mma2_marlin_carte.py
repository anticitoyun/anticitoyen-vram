"""C17 sur carte (saute sans CUDA) : `nvfp4_gemm_grouped_mma(..., marlin=decal)` sur la
disposition Marlin (preparer_pile : repack CUDA + échelles S0E5M3) contre le même noyau sur
la pile naturelle — mêmes xq/xsf/tuiles. Scellé (poste7-c17-scelle-mesure1-ter § 1) : au bit
sur les lignes dont aucune échelle n'est annulée par le repack, ≤ 2 ulp fp32 par ligne ailleurs ;
témoin cassant : un decal faux rend une sortie différente."""
import os

import pytest
import torch

from acvram.kernels import get_extension
from acvram.kernels import marlin_port as MP

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _pile(E, N, K, seed, dev):
    g = torch.Generator().manual_seed(seed)
    qw = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8)
    e = torch.randint(2, 12, (E, N, K // 16), generator=g); m = torch.randint(0, 8, (E, N, K // 16), generator=g)
    bs = ((e << 3) | m).to(torch.uint8)
    bs[:, 0, 0] = 0x7E                                                    # 448 : facteur 1
    bs[torch.rand(E, N, K // 16, generator=g) < 0.001] = 3                # sous-normales : annulées par le repack
    gs = (torch.rand(E, generator=g) * 0.01 + 0.001)
    return qw.to(dev), bs.to(dev), gs.to(dev)


@pytest.mark.parametrize("E,N,K,G", [(8, 256, 512, 96), (16, 768, 2048, 96)])
def test_mma2_sur_marlin_egal_au_naturel_hors_echelles_annulees(E, N, K, G):
    ext = get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("MMA FP4 indisponible")
    dev = torch.device("cuda:0")
    qw, bs, gs = _pile(E, N, K, 3, dev)
    w_m, s_m, g_m = MP.preparer_pile(qw, bs, gs)
    facteur = gs.float()[0].item() * 2.0 ** 119 / g_m[0].item()
    import math
    decal = 15 + int(round(math.log2(facteur)))
    # activations : G lignes réparties sur les experts, tuiles de 16
    torch.manual_seed(5)
    eid = torch.randint(0, E, (G,), device=dev); ordre = torch.argsort(eid, stable=True); es = eid[ordre]
    cnt = torch.bincount(es, minlength=E)
    from acvram.engine.model import MoEBlock, _qa_compteurs
    tiles = MoEBlock._tuiles(cnt, 16)
    xs = (torch.randn(G, K, device=dev) * 0.5).to(torch.bfloat16)
    xq, xsf, gr = ext.nvfp4_quant_act(xs, None, None, _qa_compteurs(dev), 0)
    ar = torch.arange(E, dtype=torch.int64)
    tq_n = (qw.data_ptr() + ar * qw.stride(0)).to(dev); tb_n = (bs.data_ptr() + ar * bs.stride(0)).to(dev)
    tq_m = (w_m.data_ptr() + ar * w_m.stride(0) * 4).to(dev); tb_m = (s_m.data_ptr() + ar * s_m.stride(0)).to(dev)
    y_n = ext.nvfp4_gemm_grouped_mma(tq_n, tb_n, gs.float().contiguous(), xq, xsf, *tiles, N, K, 16, 4, 128, gr)
    y_m = ext.nvfp4_gemm_grouped_mma(tq_m, tb_m, gs.float().contiguous(), xq, xsf, *tiles, N, K, 16, 4, 128, gr, decal)
    torch.cuda.synchronize()
    annulees = ((bs.view(torch.float8_e4m3fn).float() * facteur * 128) < 2)          # [E, N, K/16]
    lignes_touchees = annulees.any(-1)                                                 # [E, N] colonnes de sortie
    touche = lignes_touchees[es]                                                       # [G, N]
    egal = (y_n == y_m) | (y_n.isnan() & y_m.isnan())
    assert bool(egal[~touche].all()), f"au bit hors échelles annulées : {int((~egal[~touche]).sum())} écarts"
    ecart = (y_n.float() - y_m.float()).abs()
    amp = y_n.float().abs().amax(1, keepdim=True).clamp_min(1e-6)
    assert bool((ecart[touche] <= 2 * 2 ** -8 * amp.expand_as(ecart)[touche]).all())    # ≤ 2 ulp bf16 sur les colonnes touchées
    assert 0 < touche.float().mean() < 0.2
    # témoin cassant : decal faux → sortie différente
    y_f = ext.nvfp4_gemm_grouped_mma(tq_m, tb_m, gs.float().contiguous(), xq, xsf, *tiles, N, K, 16, 4, 128, gr, decal + 1)
    assert not torch.equal(y_f, y_m)


def test_bloc_moe_decode_prend_mma2_sur_marlin(monkeypatch):
    """Sous la disposition unique (pile rendue), ACVRAM_MOE_DECODE_MMA_MARLIN=1 envoie le
    décodage (t ≥ MIN_T) sur mma2-Marlin (`_forward_grouped_mma`, chemin compté
    `decode_mma_marlin`) ; sortie sous le juge fp32 par ligne comme la GEMV Marlin ; à
    0 (défaut) `_forward_grouped_mma` rend None et la GEMV Marlin sert."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from conftest import attendre_chemin
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    from test_gemv_marlin import _hors_par_ligne, TOL_HORS
    from acvram.engine import model as MD
    ext = get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemm_grouped_mma") or MP.charger(compiler=False) is None:
        pytest.skip("MMA FP4 ou extension Marlin indisponible")
    dev = torch.device("cuda:0")
    E, H, I, top_k, T = 8, 256, 128, 2, 12
    bloc = _bloc_moe_jouet(E, H, I, top_k)
    monkeypatch.setattr(MD, "_GEMV_LAYOUT", "marlin")
    assert bloc._try_build_stacks() and bloc._stacks_marlin is not None and bloc._stacks["gate_proj"][1] is None
    torch.manual_seed(T)
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    topw, topi = torch.topk(torch.softmax(bloc.router(x).float(), -1), top_k, dim=-1)
    topw = (topw / topw.sum(-1, keepdim=True)).float()
    from acvram.quant.nvfp4 import dequantize_nvfp4 as dq
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for j in range(top_k):
            e = int(topi[t, j]); m = bloc.experts[e]
            wg, wu, wd = (dq(getattr(m, n).qweight, torch.float32).to(dev) for n in ("gate_proj", "up_proj", "down_proj"))
            a = torch.nn.functional.silu(x[t].float() @ wg.T) * (x[t].float() @ wu.T)
            ref[t] += topw[t, j] * (a.to(torch.bfloat16).float() @ wd.T)
    monkeypatch.setattr(MD, "_MOE_DECODE_MMA_MARLIN", False)
    assert bloc._forward_grouped_mma(x, topw, topi.to(torch.int32)) is None       # défaut : pile rendue, None
    monkeypatch.setattr(MD, "_MOE_DECODE_MMA_MARLIN", True)
    y = bloc._forward_grouped_mma(x, topw, topi.to(torch.int32))
    assert y is not None
    attendre_chemin(bloc, "decode_mma_marlin")
    yb = bloc._forward_grouped(x, topw, topi.to(torch.int32))                     # GEMV Marlin, même disposition
    attendre_chemin(bloc, "gemv_marlin")
    h, hb = _hors_par_ligne(y.float(), ref), _hors_par_ligne(yb.float(), ref)
    assert h <= TOL_HORS * ref.numel() and hb <= TOL_HORS * ref.numel(), (h, hb)
