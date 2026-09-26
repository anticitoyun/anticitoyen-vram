"""GEMV groupée v2 (`nvfp4_gemv_grouped_v2`, `_gateup_v2`) : paires triées
par expert, poids lus une fois pour ≤ 4 jetons du même expert — contre v1,
IDENTIQUE AU BIT (même arithmétique, même ordre : `nvfp4_row_dot_warp`
dupliquée par jeton). Routages : aléatoire Coder (b=12, top_k 8, 128
experts), un expert qui reçoit tous les jetons (sous-segments > TPB),
créneaux fantômes (eid = −1). Bras cassant : `ordre` faux → différent.
Carte requise (noyaux CUDA)."""
import pytest
import torch

from acvram.kernels import get_extension
from acvram.quant.nvfp4 import quantize_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or get_extension() is None
                                or not hasattr(get_extension(), "nvfp4_gemv_grouped_v2"),
                                reason="carte et extension avec v2 requises")
DEV = "cuda"


def _pile(E, M, K, seed):
    g = torch.Generator().manual_seed(seed)
    ts = [quantize_nvfp4((torch.randn(M, K, generator=g) * 0.05).to(torch.bfloat16)) for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous().to(DEV)
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous().to(DEV)
    gs = torch.stack([t.global_scale.float().reshape(()) for t in ts]).contiguous().to(DEV)
    return qw, bs, gs


def _routage(b, k, E, seed, mode="aleatoire"):
    g = torch.Generator().manual_seed(seed)
    if mode == "un_expert":
        topi = torch.full((b, k), 3); topi[:, 1:] = torch.stack([torch.randperm(E, generator=g)[:k - 1] for _ in range(b)])
    else:                                              # aléatoire, fantômes
        topi = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(b)])
    eid = topi.reshape(-1).to(torch.int32)
    if mode == "fantomes":
        eid[-2 * k:] = -1                              # deux jetons fantômes
    tok = torch.arange(b, dtype=torch.int32).repeat_interleave(k)
    return eid.to(DEV), tok.to(DEV)


def _tri(eid):
    ordre = torch.argsort(eid, stable=True).to(torch.int32)
    return eid[ordre.long()].contiguous(), ordre


@pytest.mark.parametrize("mode", ["aleatoire", "un_expert", "fantomes"])
@pytest.mark.parametrize("K,M", [(2048, 768), (768, 2048)])        # gate/up et down de Coder
def test_v2_egale_v1_au_bit(mode, K, M):
    ext = get_extension()
    E, b, k = 32, 12, 8
    qw, bs, gs = _pile(E, M, K, seed=K + M)
    eid, tok = _routage(b, k, E, seed=7, mode=mode)
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped(qw, bs, gs, eid, tok, x, K)
    eid_s, ordre = _tri(eid)
    y2 = ext.nvfp4_gemv_grouped_v2(qw, bs, gs, eid_s, tok[ordre.long()].contiguous(), ordre, x, K)
    assert torch.equal(y1, y2), int((y1 != y2).sum())
    if mode == "fantomes":
        assert torch.equal(y2[-2 * k:], torch.zeros_like(y2[-2 * k:]))


def test_gateup_v2_egale_gateup_v1_au_bit():
    ext = get_extension()
    E, b, k, K, M = 32, 12, 8, 2048, 768
    qg, bg, gsg = _pile(E, M, K, seed=1)
    qu, bu, gsu = _pile(E, M, K, seed=2)
    eid, tok = _routage(b, k, E, seed=9, mode="un_expert")
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped_gateup(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)
    eid_s, ordre = _tri(eid)
    y2 = ext.nvfp4_gemv_grouped_gateup_v2(qg, bg, gsg, qu, bu, gsu, eid_s, tok[ordre.long()].contiguous(), ordre, x, K, 0)
    assert torch.equal(y1, y2), int((y1 != y2).sum())


def test_bras_cassant_ordre_faux():
    ext = get_extension()
    E, b, k, K, M = 32, 12, 8, 2048, 768
    qw, bs, gs = _pile(E, M, K, seed=3)
    eid, tok = _routage(b, k, E, seed=11)
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped(qw, bs, gs, eid, tok, x, K)
    eid_s, ordre = _tri(eid)
    faux = torch.roll(ordre, 1)
    y2 = ext.nvfp4_gemv_grouped_v2(qw, bs, gs, eid_s, tok[ordre.long()].contiguous(), faux, x, K)
    assert not torch.equal(y1, y2), "une place d'origine fausse doit se voir"


def test_le_bloc_moe_decode_v2_egale_v1(monkeypatch):
    """Intégration : `_forward_grouped` sous ACVRAM_MOE_GEMV=v2 contre v1 sur
    un bloc réel n'est possible qu'avec un modèle chargé — couvert par
    ppl-decode-kv en situ (poste3). Ici : le tri est stable et déterministe."""
    eid = torch.tensor([5, 2, 5, -1, 2, 7], dtype=torch.int32, device=DEV)
    eid_s, ordre = _tri(eid)
    assert eid_s.tolist() == [-1, 2, 2, 5, 5, 7] and ordre.tolist() == [3, 1, 4, 0, 2, 5]


# ---- x en registres (poste7-gemv-experts-dernier-geste-18-09) ------------------
XREG = pytest.mark.skipif(not torch.cuda.is_available() or get_extension() is None
                          or not hasattr(get_extension(), "nvfp4_gemv_grouped_xreg"),
                          reason="extension sans xreg")


@XREG
@pytest.mark.parametrize("mode", ["aleatoire", "un_expert", "fantomes"])
@pytest.mark.parametrize("K,M", [(2048, 768), (768, 2048), (1024, 512)])   # 2 uint4 par voie, 24 (reste), 1
def test_xreg_egale_v1_au_bit(mode, K, M):
    """x en registres par tranche + LDS.128 : mêmes expressions que
    nvfp4_row_dot_warp → identique au bit à v1, gate/up et down."""
    ext = get_extension()
    E, b, k = 32, 12, 8
    qw, bs, gs = _pile(E, M, K, seed=K + M + 1)
    eid, tok = _routage(b, k, E, seed=5, mode=mode)
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped(qw, bs, gs, eid, tok, x, K)
    y3 = ext.nvfp4_gemv_grouped_xreg(qw, bs, gs, eid, tok, x, K)
    assert torch.equal(y1, y3), int((y1 != y3).sum())


@XREG
def test_gateup_xreg_egale_gateup_v1_au_bit():
    ext = get_extension()
    E, b, k, K, M = 32, 12, 8, 2048, 768
    qg, bg, gsg = _pile(E, M, K, seed=21)
    qu, bu, gsu = _pile(E, M, K, seed=22)
    eid, tok = _routage(b, k, E, seed=13)
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped_gateup(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)
    y3 = ext.nvfp4_gemv_grouped_gateup_xreg(qg, bg, gsg, qu, bu, gsu, eid, tok, x, K, 0)
    assert torch.equal(y1, y3), int((y1 != y3).sum())


@XREG
def test_bras_cassant_xreg_x_decale():
    """Le bras qui doit différer : la même GEMV sur x décalé d'un flottant."""
    ext = get_extension()
    E, b, k, K, M = 32, 12, 8, 2048, 768
    qw, bs, gs = _pile(E, M, K, seed=31)
    eid, tok = _routage(b, k, E, seed=17)
    x = torch.randn(b, K, device=DEV).to(torch.bfloat16)
    y1 = ext.nvfp4_gemv_grouped(qw, bs, gs, eid, tok, x, K)
    y3 = ext.nvfp4_gemv_grouped_xreg(qw, bs, gs, eid, tok, torch.roll(x, 1, dims=1).contiguous(), K)
    assert not torch.equal(y1, y3)
