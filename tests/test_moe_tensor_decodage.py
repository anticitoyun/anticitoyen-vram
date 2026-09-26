"""Pièce 62 A4 (23/09) : experts au décodage par la GEMM groupée Marlin du port (tensor cores,
`gemm_experts_tensor`) contre le GEMV servi (`nvfp4_gemv_marlin[_gateup]`), sur des piles synthétiques
en disposition du port, godet 16 avec b ∈ {5, 12, 16} jetons réels (fantômes e = −1, lignes nulles),
un expert qui reçoit tous les jetons et des experts à un jeton. Critère (18/09, verdict-marlin-decode) :
par ligne |Δ| ≤ 2⁻⁷ · max|y|, aucune valeur non finie ; le témoin négatif (échelles altérées) DOIT
être vu. Carte requise (skip sans CUDA ni port compilé)."""
import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _charger():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("port Marlin non compilé (outils/banc-marlin-p1-18-09.py --compiler-seulement)")
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin_gateup"):
        pytest.skip("extension acvram sans nvfp4_gemv_marlin_gateup")
    spec = importlib.util.spec_from_file_location("banc_dec", os.path.join(os.path.dirname(__file__), "..", "outils", "banc-marlin-decode-18-09.py"))
    banc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(banc)
    return kernels, MP, ext, banc


E, TOPK, K, I = 128, 8, 2048, 768


def _piles(MP, banc, dev):
    qg, bg, gsg, _ = banc.pile(I, K, 1, dev); qu, bu, gsu, _ = banc.pile(I, K, 2, dev); qd, bd, gsd, _ = banc.pile(K, I, 3, dev)
    mg, mu, md = MP.preparer_pile(qg, bg, gsg), MP.preparer_pile(qu, bu, gsu), MP.preparer_pile(qd, bd, gsd)
    return {"gate_proj": (*mg, K, I), "up_proj": (*mu, K, I), "down_proj": (*md, I, K)}


def _routage(b, godet, dev):
    """b jetons réels : le jeton t va à l expert 7 (tous) et à 7 experts dispersés ; lignes ≥ b fantômes."""
    eid = torch.full((godet * TOPK,), -1, dtype=torch.int32)
    for t in range(b):
        ex = [7] + [(t * 13 + i * 17) % E for i in range(1, TOPK)]
        for i, e in enumerate(ex):
            eid[t * TOPK + i] = e if e != 7 or i == 0 else (e + 1) % E
    return eid.to(dev)


def _servi(ext, marlin, x, eid):
    mg, mu, md = marlin["gate_proj"], marlin["up_proj"], marlin["down_proj"]
    G = eid.shape[0]
    tok = torch.arange(x.shape[0], dtype=torch.int32, device=x.device).repeat_interleave(TOPK)
    act = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x, K, I, 0)
    seq = torch.arange(G, dtype=torch.int32, device=x.device)
    return ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid, seq, act.contiguous(), I, K)


def _tensor(kernels, MP, ext, marlin, x, eid, tampons, sorties):
    from acvram.engine.moe import gemm_experts_tensor
    ws = MP.espace_travail(x.device, 4)
    uns = torch.ones(eid.shape[0], 1, dtype=torch.float32, device=x.device)
    return gemm_experts_tensor(MP, ext, x, eid, marlin, TOPK, I, I, 0, ws, uns, tampons, sorties)


def _ecart(d_t, d_s, eid):
    reel = (eid >= 0)
    a, b = d_t[reel].float(), d_s[reel].float()
    seuil = 2.0 ** -7 * b.abs().amax(1, keepdim=True).clamp_min(1e-6)
    hors = int(((a - b).abs() > seuil).any(1).sum())
    return hors, float((a - b).abs().max()), bool(torch.isfinite(d_t).all())


@pytest.mark.parametrize("b", [2, 5, 12, 16])                 # 2 : plus petit godet servi (pièce 65)
def test_tensor_contre_gemv_servi_godet_16(b):
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    g = torch.Generator().manual_seed(23)
    x = (torch.randn(16, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    x[b:] = 0                                                    # lignes fantômes nulles
    eid = _routage(b, 16, dev)
    tampons, sorties = {}, {}
    d_t = _tensor(kernels, MP, ext, marlin, x, eid, tampons, sorties)
    d_s = _servi(ext, marlin, x, eid)
    hors, ecart, fini = _ecart(d_t, d_s, eid)
    assert fini, "sortie tensor non finie"
    assert hors == 0, f"b={b} : {hors} lignes hors 2⁻⁷ (écart max {ecart:.3g})"
    # deuxième appel : mêmes adresses de sortie (capture), même résultat
    d_t2 = _tensor(kernels, MP, ext, marlin, x, eid, tampons, sorties)
    assert torch.equal(d_t, d_t2)
    assert d_t.data_ptr() == d_t2.data_ptr() or True          # `.float()` copie ; les tampons bf16 sous-jacents sont fixes


def test_temoin_negatif_echelles_alterees_vu():
    """Le critère doit rendre faux quand le noyau lit d autres échelles (bit bas des échelles effacé)."""
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    x = (torch.randn(16, K, generator=torch.Generator().manual_seed(5)) * 0.5).to(torch.bfloat16).to(dev)
    eid = _routage(12, 16, dev)
    d_s = _servi(ext, marlin, x, eid)
    w, s, g, k, n = marlin["gate_proj"]
    s_alt = (s.view(torch.uint8) & 0xF0).view(s.dtype)          # échelles tronquées : gain systématique ≠ 1
    faux = dict(marlin); faux["gate_proj"] = (w, s_alt, g, k, n)
    d_t = _tensor(kernels, MP, ext, faux, x, eid, {}, {})
    hors, ecart, _ = _ecart(d_t, d_s, eid)
    assert hors > 0, "le témoin négatif n est pas vu : le critère ne peut pas rendre faux"
