"""Pièce 209 (b) : échelle globale Marlin par (expert, colonne) [E, N] — dans l'épilogue du GEMM MoE porté
(`gs_par_colonne`) et dans le GEMV Marlin CUDA (`gs_ld`). Juges : (1) une pile SANS écrasement préparée au scalaire puis
« forcée » par colonne (g[e] recopié sur N colonnes) rend AU BIT les mêmes sorties — chemin tensor (godet 8 et 16),
GEMV une projection, GEMV w13 (moitié d'up en VUE de stride 2N) ; (2) les piles RÉELLES du Coder (couche 0 gate/up, à
facteur par ligne) : dépaquetage Triton [E, N] au bit du torch et de la référence, GEMM tensor et GEMV servi contre la
référence fp32 des poids dépaquetés (|Δ| ≤ 2⁻⁷·max|y| par ligne) ; (3) une pile sans écrasement garde g [E] (au bit d'avant).
Carte requise."""
import importlib.util
import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
E, TOPK, K, I = 128, 8, 2048, 768
ALIAS = os.environ.get("ACVRAM_ALIAS_CODER", "/mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c")
E4 = torch.float8_e4m3fn


def _charger():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("port Marlin non compilé")
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemv_marlin_w13"):
        pytest.skip("extension acvram absente")
    spec = importlib.util.spec_from_file_location("banc_dec", os.path.join(os.path.dirname(__file__), "..", "outils", "banc-marlin-decode-18-09.py"))
    banc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(banc)
    return kernels, MP, ext, banc


def _routage(b, godet, dev, n_exp=E):
    eid = torch.full((godet * TOPK,), -1, dtype=torch.int32)
    for t in range(b):
        ex = [7 % n_exp] + [(t * 13 + i * 17) % n_exp for i in range(1, TOPK)]
        for i, e in enumerate(ex):
            eid[t * TOPK + i] = e if e != 7 % n_exp or i == 0 else (e + 1) % n_exp
    return eid.to(dev)


def _tensor(MP, ext, marlin, x, eid, n_gate=I, k_down=I):
    from acvram.engine.moe import gemm_experts_tensor
    ws = MP.espace_travail(x.device, 4)
    uns = torch.ones(eid.shape[0], 1, dtype=torch.float32, device=x.device)
    return gemm_experts_tensor(MP, ext, x, eid, marlin, TOPK, n_gate, k_down, 0, ws, uns, {}, {})


def _gemv(ext, marlin, x, eid, n_gate=I, k_down=I, kk=K):
    """Le chemin servi à b=1 : w13 si présent, sinon gate‖up, puis down."""
    G = eid.shape[0]
    tok = torch.arange(x.shape[0], dtype=torch.int32, device=x.device).repeat_interleave(TOPK)
    w13 = marlin.get("w13")
    if w13 is not None:
        act = ext.nvfp4_gemv_marlin_w13(w13[0], w13[1], w13[2], w13[3], eid, tok, x, kk, n_gate, 0, None)[:, :n_gate]
    else:
        mg, mu = marlin["gate_proj"], marlin["up_proj"]
        act = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x, kk, n_gate, 0, None)
    md = marlin["down_proj"]
    seq = torch.arange(G, dtype=torch.int32, device=x.device)
    return ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid, seq, act.contiguous(), k_down, kk, None)


def _w13(marlin, par_colonne):
    wg, sg, gg, k, m = marlin["gate_proj"]; wu, su, gu, _, _ = marlin["up_proj"]
    w13, s13 = torch.cat([wg, wu], dim=2).contiguous(), torch.cat([sg, su], dim=2).contiguous()
    if par_colonne:                                      # la construction de MoEBlock._construire_marlin (209)
        ng = s13.shape[2] // 2
        g13 = torch.cat([gg.reshape(-1, 1).expand(-1, ng) if gg.dim() == 1 else gg,
                         gu.reshape(-1, 1).expand(-1, ng) if gu.dim() == 1 else gu], dim=1).contiguous()
        return (w13, s13, g13, g13[:, ng:], k, m, None, None)
    return (w13, s13, gg, gu, k, m, (gu / gg).contiguous(), torch.ones_like(gg))


def _forcer_colonne(marlin, n):
    """Même pile, g[e] recopié sur n colonnes : la sortie doit être au bit de la version scalaire."""
    out = {}
    for nom, pile in marlin.items():
        if nom == "w13":
            continue
        w, s, g, k, m = pile
        out[nom] = (w, s, g.reshape(-1, 1).expand(-1, n if nom != "down_proj" else K).contiguous(), k, m)
    return out


def _ecart(a, b, eid):
    reel = eid >= 0
    a, b = a[reel].float(), b[reel].float()
    seuil = 2.0 ** -7 * b.abs().amax(1, keepdim=True).clamp_min(1e-6)
    return int(((a - b).abs() > seuil).any(1).sum()), float((a - b).abs().max())


@pytest.mark.parametrize("b,godet", [(1, 1), (8, 8), (16, 16)])
def test_par_colonne_forcee_au_bit_du_scalaire(b, godet):
    kernels, MP, ext, banc = _charger()
    dev = torch.device("cuda")
    qg, bg, gsg, _ = banc.pile(I, K, 1, dev); qu, bu, gsu, _ = banc.pile(I, K, 2, dev); qd, bd, gsd, _ = banc.pile(K, I, 3, dev)
    assert MP.echelles_ecrasees(bg) == 0 and MP.echelles_ecrasees(bu) == 0
    mg, mu, md = MP.preparer_pile(qg, bg, gsg), MP.preparer_pile(qu, bu, gsu * 1.37), MP.preparer_pile(qd, bd, gsd)
    assert mg[2].shape == (E,), "pile sans écrasement : g [E], préparation d'avant"
    marlin = {"gate_proj": (*mg, K, I), "up_proj": (*mu, K, I), "down_proj": (*md, I, K)}
    marlin["w13"] = _w13(marlin, False)
    forcee = _forcer_colonne(marlin, I); forcee["w13"] = _w13(forcee, True)
    assert forcee["w13"][2].shape == (E, 2 * I) and forcee["w13"][3].stride(0) == 2 * I
    x = (torch.randn(godet, K, generator=torch.Generator().manual_seed(b)) * 0.5).to(torch.bfloat16).to(dev)
    eid = _routage(b, godet, dev)
    if godet >= 8:
        # tensor SANS w13 (gate et up séparées, chacune avec son g) : par colonne forcée = scalaire AU BIT (même flottant
        # multiplié au même endroit de l'épilogue, branches m_block_size_8 (godet 8) et 16)
        sans = {k: v for k, v in marlin.items() if k != "w13"}; sans_f = {k: v for k, v in forcee.items() if k != "w13"}
        d1, d2 = _tensor(MP, ext, sans, x, eid), _tensor(MP, ext, sans_f, x, eid)
        assert torch.equal(d1, d2), f"chemin tensor godet {godet} : {int((d1 != d2).sum())} valeurs diffèrent"
        # tensor w13 par colonne : up prend directement son g dans l'épilogue (un arrondi bf16 de moins que le w13
        # scalaire, qui corrige up par gu/gg dans moe_act — 82 ter : « au 2⁻⁷, pas au bit ») : juge 2⁻⁷ par ligne
        # contre le chemin séparé, comme test_moe_w13
        d3 = _tensor(MP, ext, forcee, x, eid)
        hors, dmax = _ecart(d3, d1, eid)
        assert hors == 0, f"w13 par colonne contre séparé, godet {godet} : {hors} lignes hors 2⁻⁷ (Δ max {dmax:.3g})"
    y1, y2 = _gemv(ext, marlin, x, eid), _gemv(ext, forcee, x, eid)
    assert torch.equal(y1, y2), f"GEMV w13 + down : {int((y1 != y2).sum())} valeurs diffèrent"
    sans = {k: v for k, v in marlin.items() if k != "w13"}; sans_f = {k: v for k, v in forcee.items() if k != "w13"}
    assert torch.equal(_gemv(ext, sans, x, eid), _gemv(ext, sans_f, x, eid)), "GEMV gate‖up séparés"


def _pile_reelle(couche, mat, dev, experts):
    from safetensors import safe_open
    from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4
    m = json.load(open(os.path.join(ALIAS, "acvram_manifest.json"))); wm = m["weight_map"]; lect = {}

    def lire(nom):
        f = wm[nom]
        if f not in lect:
            lect[f] = safe_open(os.path.join(ALIAS, f), "pt", device="cpu")
        return lect[f].get_tensor(nom)
    base = f"model.layers.{couche}.mlp.experts.{{}}.{mat}.weight"
    ts = [NVFP4Tensor(lire(base.format(e) + ".qweight"), lire(base.format(e) + ".block_scale").view(E4),
                      lire(base.format(e) + ".global_scale").float(), None, None) for e in experts]
    for t in ts:
        t.shape = (t.qweight.shape[0], t.qweight.shape[1] * 2); t.padded_in = t.shape[1]
    qw = torch.stack([t.qweight for t in ts]).to(dev); bs = torch.stack([t.block_scale for t in ts]).to(dev)
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).float().to(dev)
    ref = torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts]).to(dev)
    return qw, bs, gs, ref


@pytest.mark.skipif(not os.path.isdir(ALIAS), reason="alias Coder absent")
def test_pile_reelle_du_coder_par_colonne_contre_la_reference():
    kernels, MP, ext, banc = _charger()
    from acvram.kernels import marlin_port as MPm
    dev = torch.device("cuda")
    m = json.load(open(os.path.join(ALIAS, "acvram_manifest.json")))
    # experts touchés de la couche 0 (gate) + 2 témoins, d'après les échelles
    from safetensors import safe_open
    wm = m["weight_map"]; base = "model.layers.0.mlp.experts.{}.gate_proj.weight.block_scale"
    lect = {}

    def lire(nom):
        f = wm[nom]
        if f not in lect:
            lect[f] = safe_open(os.path.join(ALIAS, f), "pt", device="cpu")
        return lect[f].get_tensor(nom)
    bs_tous = torch.stack([lire(base.format(e)).view(E4) for e in range(E)])
    touches = [e for e in range(E) if MP.echelles_ecrasees(bs_tous[e:e + 1]) > 0]
    experts = touches + [e for e in range(E) if e not in touches][:2]
    ne = len(experts)
    qg, bg, gsg, Wg = _pile_reelle(0, "gate_proj", dev, experts)
    qu, bu, gsu, Wu = _pile_reelle(0, "up_proj", dev, experts)
    qd, bd, gsd, Wd = _pile_reelle(0, "down_proj", dev, experts)
    assert MP.echelles_ecrasees(bg) > 0 and MP.echelles_ecrasees(bd) == 0
    mg, mu, md = MP.preparer_pile(qg, bg, gsg), MP.preparer_pile(qu, bu, gsu), MP.preparer_pile(qd, bd, gsd)
    assert mg[2].shape == (ne, I) and mu[2].shape == (ne, I) and md[2].shape == (ne,)
    # dépaquetage : Triton [E, N] au bit du torch et de la référence
    Wt = MP.depaqueter_marlin(mg[0], mg[1], mg[2], K, I, noyau="triton")
    assert torch.equal(Wt, Wg), f"dépaquetage Triton par colonne : {int((Wt != Wg).sum())} valeurs fausses"
    marlin = {"gate_proj": (*mg, K, I), "up_proj": (*mu, K, I), "down_proj": (*md, I, K)}
    marlin["w13"] = _w13(marlin, True)
    torch.manual_seed(2090)
    for b, godet in ((1, 1), (8, 8)):
        x = (torch.randn(godet, K, device=dev) * 0.5).to(torch.bfloat16)
        eid = _routage(b, godet, dev, n_exp=ne)
        tok = torch.arange(godet, device=dev).repeat_interleave(TOPK)
        reel = eid >= 0
        xe = x[tok[reel]].float(); ee = eid[reel].long()
        g = torch.einsum("gk,gnk->gn", xe, Wg[ee].float()); u = torch.einsum("gk,gnk->gn", xe, Wu[ee].float())
        act = torch.nn.functional.silu(g) * u
        d_ref = torch.einsum("gi,gki->gk", act, Wd[ee].float())
        y = _gemv(ext, marlin, x, eid)
        hors, dmax = _ecart(y, torch.zeros(eid.shape[0], K, device=dev).index_copy_(0, reel.nonzero().flatten(), d_ref), eid)
        assert hors == 0, f"GEMV b={b} : {hors} lignes hors 2⁻⁷ (Δ max {dmax:.3g})"
        if godet >= 8:
            d = _tensor(MP, ext, marlin, x, eid)
            hors, dmax = _ecart(d, torch.zeros(eid.shape[0], K, device=dev).index_copy_(0, reel.nonzero().flatten(), d_ref), eid)
            assert hors == 0, f"tensor godet {godet} : {hors} lignes hors 2⁻⁷ (Δ max {dmax:.3g})"


def test_mma2_sur_marlin_refuse_une_pile_par_ligne(monkeypatch):
    """C17 (`ACVRAM_MOE_DECODE_MMA_MARLIN=1`, mma2 sur les tuiles Marlin à échelle naturelle + décalage global) n'a pas de
    décalage unique pour une pile à g [E, N] : `_forward_grouped_mma` rend None (GEMV Marlin gardée) et `_decal_marlin` lève."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    from acvram.engine import moe as MOE_D
    kernels, MP, ext, banc = _charger()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma"):
        pytest.skip("MMA FP4 indisponible")
    dev = torch.device("cuda:0")
    E_, H, I_, top_k, T = 8, 256, 128, 2, 12
    bloc = _bloc_moe_jouet(E_, H, I_, top_k)
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin"); monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MOE_W13", False); monkeypatch.setattr(MOE_D, "_MOE_DECODE_MMA_MARLIN", True)
    assert bloc._try_build_stacks()                   # sous la disposition Marlin : piles Marlin construites, naturelle rendue
    assert bloc._stacks_marlin is not None and bloc._stacks["gate_proj"][1] is None
    # la même pile, forcée par colonne (g[e] recopié) : ce que rend preparer_pile pour une pile à sous-normales
    for n in ("gate_proj", "up_proj", "down_proj"):
        w, sc, g, k, m = bloc._stacks_marlin[n]
        bloc._stacks_marlin[n] = (w, sc, g.reshape(-1, 1).expand(-1, w.shape[2] // 2).contiguous(), k, m)
    torch.manual_seed(T)
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    topw, topi = torch.topk(torch.softmax(bloc.router(x).float(), -1), top_k, dim=-1)
    assert bloc._forward_grouped_mma(x, (topw / topw.sum(-1, keepdim=True)).float(), topi.to(torch.int32)) is None
    with pytest.raises(RuntimeError, match="par \\(expert, colonne\\)"):
        bloc._decal_marlin("gate_proj")
