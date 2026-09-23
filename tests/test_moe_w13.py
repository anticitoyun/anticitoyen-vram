"""Pièce 82 (23/09) : gate·up fusionnés en une pile Marlin w13 (`ACVRAM_MOE_W13`).

1. GEMV (godets < MOE_TENSOR_MIN_T, b = 1) : `nvfp4_gemv_marlin_w13` lit gate et up dans w13 avec leurs échelles
   globales propres — AU BIT de `nvfp4_gemv_marlin_gateup` sur les deux piles séparées.
2. Chemin tensor : une GEMM w13 à l échelle de gate, up corrigée dans moe_act — par ligne |Δ| ≤ 2⁻⁷ · max|y| contre
   le chemin gate/up séparé, et le témoin négatif (rapport d échelles oublié) DOIT être vu.
3. Disposition : w13 = gate ‖ up ligne de tuiles par ligne de tuiles ; la découpe rend les piles d origine.
Carte requise (skip sans CUDA, port ou extension à jour)."""
import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")

E, TOPK, K, I = 128, 8, 2048, 768


def _charger():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("port Marlin non compilé")
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin_w13"):
        pytest.skip("extension acvram sans nvfp4_gemv_marlin_w13 (à recompiler)")
    spec = importlib.util.spec_from_file_location("banc_dec", os.path.join(os.path.dirname(__file__), "..", "outils", "banc-marlin-decode-18-09.py"))
    banc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(banc)
    return kernels, MP, ext, banc


def _piles(MP, banc, dev):
    """Échelles globales de gate et d up DISTINCTES (graines différentes), comme sur l alias servi (12/12, 71 bis)."""
    qg, bg, gsg, _ = banc.pile(I, K, 1, dev); qu, bu, gsu, _ = banc.pile(I, K, 2, dev); qd, bd, gsd, _ = banc.pile(K, I, 3, dev)
    mg, mu, md = MP.preparer_pile(qg, bg, gsg), MP.preparer_pile(qu, bu, gsu * 1.37), MP.preparer_pile(qd, bd, gsd)
    marlin = {"gate_proj": (*mg, K, I), "up_proj": (*mu, K, I), "down_proj": (*md, I, K)}
    assert not torch.equal(mg[2], mu[2]), "le test exige des échelles globales gate/up distinctes"
    return marlin


def _w13(marlin):
    """La construction de `MoEBlock._construire_marlin` sous ACVRAM_MOE_W13=1, telle quelle."""
    wg, sg, gg, k, m = marlin["gate_proj"]
    wu, su, gu, _, _ = marlin["up_proj"]
    w13, s13 = torch.cat([wg, wu], dim=2).contiguous(), torch.cat([sg, su], dim=2).contiguous()
    return (w13, s13, gg, gu, k, m, (gu / gg).contiguous(), torch.ones_like(gg))


def _routage(b, godet, dev):
    eid = torch.full((godet * TOPK,), -1, dtype=torch.int32)
    for t in range(b):
        ex = [7] + [(t * 13 + i * 17) % E for i in range(1, TOPK)]
        for i, e in enumerate(ex):
            eid[t * TOPK + i] = e if e != 7 or i == 0 else (e + 1) % E
    return eid.to(dev)


def test_disposition_w13_decoupe_rend_les_piles():
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    w13 = _w13(marlin)
    assert torch.equal(w13[0][..., : 2 * I], marlin["gate_proj"][0]) and torch.equal(w13[0][..., 2 * I:], marlin["up_proj"][0])
    assert torch.equal(w13[1][..., :I], marlin["gate_proj"][1]) and torch.equal(w13[1][..., I:], marlin["up_proj"][1])


@pytest.mark.parametrize("b", [1, 3, 5, 12])
@pytest.mark.parametrize("xdt", [torch.bfloat16, torch.float32])
def test_gemv_w13_au_bit_du_gateup_separe(b, xdt):
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    w13 = _w13(marlin)
    mg, mu = marlin["gate_proj"], marlin["up_proj"]
    x = (torch.randn(b, K, generator=torch.Generator().manual_seed(b)) * 0.5).to(xdt).to(dev)
    eid = _routage(b, b, dev)
    tok = torch.arange(b, dtype=torch.int32, device=dev).repeat_interleave(TOPK)
    for act in (0, 1):
        ref = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x, K, I, act)
        out = ext.nvfp4_gemv_marlin_w13(w13[0], w13[1], w13[2], w13[3], eid, tok, x, K, I, act)
        assert torch.equal(out, ref), f"b={b} act={act} : GEMV w13 différent du GEMV séparé (max {float((out - ref).abs().max()):.3g})"


def _tensor(MP, ext, marlin, x, eid):
    from acvram.engine.moe import gemm_experts_tensor
    ws = MP.espace_travail(x.device, 4)
    uns = torch.ones(eid.shape[0], 1, dtype=torch.float32, device=x.device)
    return gemm_experts_tensor(MP, ext, x, eid, marlin, TOPK, I, I, 0, ws, uns, {}, {})


def _hors(a, b, eid):
    reel = eid >= 0
    a, b = a[reel].float(), b[reel].float()
    seuil = 2.0 ** -7 * b.abs().amax(1, keepdim=True).clamp_min(1e-6)
    return int(((a - b).abs() > seuil).any(1).sum()), float((a - b).abs().max())


@pytest.mark.parametrize("b", [8, 12, 16])
def test_tensor_w13_au_2_moins_7_du_separe(b):
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    x = (torch.randn(16, K, generator=torch.Generator().manual_seed(23)) * 0.5).to(torch.bfloat16).to(dev)
    x[b:] = 0
    eid = _routage(b, 16, dev)
    ref = _tensor(MP, ext, marlin, x, eid)
    fus = dict(marlin); fus["w13"] = _w13(marlin)
    fus["gate_proj"], fus["up_proj"] = (None, None, *marlin["gate_proj"][2:]), (None, None, *marlin["up_proj"][2:])
    out = _tensor(MP, ext, fus, x, eid)
    assert bool(torch.isfinite(out).all())
    hors, ecart = _hors(out, ref, eid)
    assert hors == 0, f"b={b} : {hors} lignes hors 2⁻⁷ (écart max {ecart:.3g})"
    # témoin négatif : sans la correction d échelle d up (rapport 1), le critère DOIT rendre faux
    faux = dict(fus); w = list(fus["w13"]); w[6] = torch.ones_like(w[6]); faux["w13"] = tuple(w)
    hors_f, _ = _hors(_tensor(MP, ext, faux, x, eid), ref, eid)
    assert hors_f > 0, "témoin négatif non vu : le critère ne peut pas rendre faux"


@pytest.mark.parametrize("t", [1, 16, 256, 3072])
def test_prefill_w13_vues_au_bit_du_separe(t):
    """Pièce 82 ter : au préfill, gate et up lues dans w13 par des VUES de colonnes (largeur stockée 2N, `ldn` du
    port) rendent les mêmes bits que les GEMM sur les piles séparées — même découpe de K (prob_n = N). Casse si
    `ldn` est ignoré (la vue lirait les mauvaises colonnes) ou si la GEMM 2N de la 82 revient au préfill."""
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    w13 = _w13(marlin)
    g0 = torch.Generator(device="cpu").manual_seed(91 + t)
    topi = torch.stack([torch.randperm(E, generator=g0)[:TOPK] for _ in range(t)]).to(dev)
    e_sorted, _ = torch.sort(topi.reshape(-1))
    G = e_sorted.numel()
    xs = (torch.randn(G, K, generator=g0) * 0.5).to(torch.bfloat16).to(dev).contiguous()
    bloc = MP.choisir_block_size(t, TOPK, E)
    s_ids, e_ids, n_post = MP.aligner_blocs(e_sorted.to(torch.int32).unsqueeze(1), bloc, E)
    ws = MP.espace_travail(dev, 4)
    uns = torch.ones(G, 1, dtype=torch.float32, device=dev)
    for nom, vue in (("gate_proj", (w13[0][:, :, :2 * I], w13[1][:, :, :I], w13[2])),
                     ("up_proj", (w13[0][:, :, 2 * I:], w13[1][:, :, I:], w13[3]))):
        assert not vue[0].is_contiguous()
        sep = MP.gemm_moe(xs, *marlin[nom][:3], s_ids, e_ids, n_post, uns, bloc, 1, G, I, K, ws)
        v = MP.gemm_moe(xs, *vue, s_ids, e_ids, n_post, uns, bloc, 1, G, I, K, ws)
        assert torch.equal(v, sep), (nom, t, (v.float() - sep.float()).abs().max().item())


def test_gemm_marlin_refuse_une_vue_d_echelles_desaccordee():
    """Une vue de B de largeur stockée 2N avec des échelles CONTIGUËS (largeur N) : refus nommé, pas une lecture fausse."""
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    w13 = _w13(marlin)
    t, G = 1, TOPK
    e_sorted = torch.arange(TOPK, dtype=torch.int32, device=dev)
    xs = torch.randn(G, K, device=dev).to(torch.bfloat16)
    s_ids, e_ids, n_post = MP.aligner_blocs(e_sorted.unsqueeze(1), 8, E)
    with pytest.raises(Exception, match="largeur stockée"):
        MP.gemm_moe(xs, w13[0][:, :, :2 * I], marlin["gate_proj"][1], w13[2], s_ids, e_ids, n_post,
                    torch.ones(G, 1, device=dev), 8, 1, G, I, K, MP.espace_travail(dev, 4))


@pytest.mark.parametrize("t", [8, 12, 32])
def test_prefill_court_w13_au_bit_du_separe(t):
    """Pièce 82 ter : un préfill court (T ≤ _MOE_GROUPED_MAX) passe par `gemm_experts_tensor` ; sous w13 et
    `w13_fusionne=False` (préfill), il doit rendre les bits du chemin séparé. Casse si la GEMM 2N y revient (82 : KL
    b=1 +0,098 sur l invite 3) ; le témoin w13_fusionne=True, lui, doit différer."""
    kernels, MP, ext, banc = _charger()
    from acvram.engine.moe import gemm_experts_tensor
    dev = torch.device("cuda", 0)
    marlin = _piles(MP, banc, dev)
    x = (torch.randn(t, K, generator=torch.Generator().manual_seed(29 + t)) * 0.5).to(torch.bfloat16).to(dev)
    eid = _routage(t, t, dev)
    fus = dict(marlin); fus["w13"] = _w13(marlin)
    fus["gate_proj"], fus["up_proj"] = (None, None, *marlin["gate_proj"][2:]), (None, None, *marlin["up_proj"][2:])
    ws = MP.espace_travail(dev, 4)
    uns = torch.ones(eid.shape[0], 1, dtype=torch.float32, device=dev)
    ref = gemm_experts_tensor(MP, ext, x, eid, marlin, TOPK, I, I, 0, ws, uns, {}, {})
    out = gemm_experts_tensor(MP, ext, x, eid, fus, TOPK, I, I, 0, ws, uns, {}, {}, w13_fusionne=False)
    assert torch.equal(out, ref), (t, (out.float() - ref.float()).abs().max().item())
    temoin = gemm_experts_tensor(MP, ext, x, eid, fus, TOPK, I, I, 0, ws, uns, {}, {}, w13_fusionne=True)
    assert not torch.equal(temoin, ref), "témoin : la GEMM 2N devrait différer du chemin séparé"

