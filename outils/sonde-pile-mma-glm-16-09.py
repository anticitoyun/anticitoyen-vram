#!/usr/bin/env python3
"""Sonde par étape de la pile MMA GLM (ordre Sage, sage-glm-pile-correctif-16-09 § 3.4).

Pile réelle (`MoEBlock._forward_grouped_mma`, MMA=1) de la couche MoE 1, sur
les 16 jetons et les 8 experts d'Océane (x réel capturé sur le mini-répertoire
bf16 de `outils/discriminateur-glm-mma0-16-09.py`), pour deux convertis :
alpha-commun (`-srcbf16-nvfp4`) et sans-AWQ (`-sansawq`). Chaque expert 0-7
est imposé à chaque jeton (poids 1/8) : même x, mêmes experts, mêmes tuiles.

Six étapes, chacune contre une référence Python de la même arithmétique
(fp64, poids déquantifiés par `dequantize_nvfp4`, x/s en fp32) :
  1 quant_act(xs[, xs2])      2 _gemm_mma gate/up      3 moe_act
  4 quant_act(act)            5 _gemm_mma down         6 moe_reduce_trie
Deux erreurs par étape, ‖a − b‖/‖b‖ poolée sur les 128 lignes :
  iso : la sortie réelle contre la référence calculée SUR LES ENTRÉES RÉELLES
        de l'étape (l'erreur propre du noyau) ;
  cum : la sortie réelle contre la référence exacte depuis x (erreur cumulée).
Pour 1 et 4 : nombre d'octets qui diffèrent de `quant_act_ref` (bit-exact
attendu) et fraction de blocs mis à zéro.

Scellé : exactement une étape où err_iso(alpha-commun)/err_iso(sans-AWQ) ≥ 3.

    outils/carte.sh python outils/sonde-pile-mma-glm-16-09.py [--jetons 16]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "tests"))

MOD = Path(_RACINE)
CONVERTIS = {
    "alpha-commun": MOD / "GLM-4.7-Flash-srcbf16-nvfp4",
    "sans-awq": MOD / "GLM-4.7-Flash-srcbf16-nvfp4-sansawq",
}
LAYER = 1
EXPERTS = list(range(8))
SORTIE = RACINE / "scratchpad" / "sonde-pile-mma-glm-16-09"


def _err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b).clamp(min=1e-30))


def _deq_act(xq: torch.Tensor, xsf: torch.Tensor, gr: torch.Tensor) -> torch.Tensor:
    """E2M1 [G, K/2] + UE4M3 [G, K/16] + g_r [G] -> fp32 [G, K], même arithmétique que le noyau."""
    from test_gemm_grouped_mma import dequant_act_ref
    return dequant_act_ref(xq, xsf, gr).float()


def _x_reel(n: int) -> torch.Tensor:
    spec = importlib.util.spec_from_file_location(
        "discr", RACINE / "outils" / "discriminateur-glm-mma0-16-09.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    if n != d.N_JETONS if hasattr(d, "N_JETONS") else False:
        raise SystemExit("--jetons doit valoir les 16 jetons d'Océane (même x)")
    if not d.CONVERTI_BF16.exists():
        d.extraire_mini(); d.convertir_bf16()
    return d.capturer_activations()                         # [16, H] fp32, CPU


def sonder(nom: str, chemin: Path, x_cpu: torch.Tensor) -> dict:
    from acvram.engine.loader import load_model
    from acvram.engine import model as M
    from acvram.quant.nvfp4 import dequantize_nvfp4
    from acvram import kernels
    from test_gemm_grouped_mma import quant_act_ref

    loaded = load_model(str(chemin), dtype=torch.bfloat16, max_model_len=64)
    mod = loaded.model.layers[LAYER].mlp
    assert isinstance(mod, M.MoEBlock), type(mod)
    if mod._stacks is None:
        mod._try_build_stacks()
    st = mod._stacks
    assert st is not None and "gate_proj" in st, mod._raison_repli
    pg, pu, pd = (st[n] for n in ("gate_proj", "up_proj", "down_proj"))
    awq = getattr(mod, "_stacks_awq", {}) or {}
    dev = pg[1].device
    ext = kernels.get_extension()

    # --- espions : mêmes fonctions, mêmes arguments, sorties relevées ---
    trace: list[tuple[str, tuple, object]] = []

    def espion(nom_fn, fn):
        def w(*a, **kw):
            r = fn(*a, **kw)
            trace.append((nom_fn, a, r))
            return r
        return w
    orig = {n: getattr(ext, n) for n in ("nvfp4_quant_act", "moe_act", "moe_reduce_trie", "moe_route_pack")}
    for n, f in orig.items():
        setattr(ext, n, espion(n, f))
    gemm0 = mod._gemm_mma
    mod._gemm_mma = espion("_gemm_mma", gemm0)

    t = x_cpu.shape[0]
    k = len(EXPERTS)
    x = x_cpu.to(dev, torch.bfloat16)
    topi = torch.tensor(EXPERTS, device=dev, dtype=torch.int32).repeat(t, 1)
    topw = torch.full((t, k), 1.0 / k, device=dev, dtype=torch.float32)
    try:
        y = mod._forward_grouped_mma(x, topw, topi)
    finally:
        for n, f in orig.items():
            setattr(ext, n, f)
        mod._gemm_mma = gemm0
    assert y is not None, "chemin MMA indisponible (repli)"
    torch.cuda.synchronize()

    # --- relevés ---
    rp = [r for n, a, r in trace if n == "moe_route_pack"]
    assert len(rp) == 1, "route_pack attendu (ACVRAM_MOE_ROUTE_PACK=1)"
    xs, ordre, inv, tw, _cnt, te, t0, tn, es, xs2 = rp[0]
    qa = [(a, r) for n, a, r in trace if n == "nvfp4_quant_act"]
    gm = [(a, r) for n, a, r in trace if n == "_gemm_mma"]
    ma = [(a, r) for n, a, r in trace if n == "moe_act"]
    rd = [(a, r) for n, a, r in trace if n == "moe_reduce_trie"]
    assert len(gm) == 3 and len(ma) == 1 and len(rd) == 1 and len(qa) in (2, 3), (len(gm), len(ma), len(rd), len(qa))
    distinct = len(qa) == 3
    (xs_in, (xq, xsf, gr)) = qa[0][0][0], qa[0][1]
    (xs2_in, (xq2, xsf2, gr2)) = (qa[1][0][0], qa[1][1]) if distinct else (xs_in, (xq, xsf, gr))
    act_in, (aq, asf, gra) = qa[-1][0][0], qa[-1][1]
    g, u, d = gm[0][1], gm[1][1], gm[2][1]
    act = ma[0][1]

    G = xs.shape[0]
    es_l = es.long()
    tok = (ordre.long() // k)
    K = pg[4]; I = pd[4]; Mo = pd[5]
    xf = x_cpu.to(dev, torch.float64)[tok]                 # [G, H] exact
    if xf.shape[1] < K:
        xf = torch.nn.functional.pad(xf, (0, K - xf.shape[1]))

    def table(nom):
        tb = awq.get(nom)
        if tb is None:
            return torch.ones(len(EXPERTS), K if nom != "down_proj" else I, device=dev, dtype=torch.float64)
        return tb[EXPERTS].double()
    s_g, s_u, s_d = table("gate_proj"), (table("up_proj") if awq.get("up_distinct") else table("gate_proj")), table("down_proj")
    # note : le noyau divise en bf16 (bf16(bf16(v)/bf16(s))) ; la référence
    # « même arithmétique » de l'étape 1 est la sortie xs du noyau route_pack
    # elle-même ; la référence exacte est x/s en fp64.
    sg_r, su_r, sd_r = s_g[es_l], s_u[es_l], s_d[es_l]    # [G, K] par ligne
    xg_ref, xu_ref = xf / sg_r, xf / su_r

    def W(pile, e):
        p = mod.experts[e]
        proj = getattr(p, {id(pg): "gate_proj", id(pu): "up_proj", id(pd): "down_proj"}[id(pile)])
        return dequantize_nvfp4(proj.qweight, torch.float32).double()   # [m, K_vrai]
    Wg = {e: W(pg, e) for e in EXPERTS}; Wu = {e: W(pu, e) for e in EXPERTS}; Wd = {e: W(pd, e) for e in EXPERTS}

    def gemm_par_expert(xin, Wt, larg):
        out = torch.zeros(G, larg, device=dev, dtype=torch.float64)
        for e in EXPERTS:
            m = es_l == e
            w = Wt[e]
            out[m, :w.shape[0]] = xin[m][:, :w.shape[1]] @ w.T
        return out

    res = {"converti": nom, "chemin": str(chemin), "up_distinct": bool(awq.get("up_distinct")),
           "tables": {n: (awq.get(n) is not None) for n in ("gate_proj", "up_proj", "down_proj")},
           "G": G, "etapes": {}}
    E = res["etapes"]

    # 1. quantification des activations gate/up (division x/s[e] fusionnée
    #    dans le noyau : les arguments relevés portent la table et e_sorted)
    def div_bf16(xin, tb, es_):
        xf_ = xin.float()
        if tb is not None:
            xf_ = (xf_ / tb[es_.long()].float()).to(torch.bfloat16).float()
        return xf_.double()
    a0 = qa[0][0]
    xq_r, xsf_r, gr_r = quant_act_ref(xs_in, a0[1], a0[2])
    deq = _deq_act(xq, xsf, gr).double()
    xs_div = div_bf16(xs_in, a0[1], a0[2])
    e1 = {"octets_diff_ref": int((xq_r != xq).sum() + (xsf_r != xsf).sum() + (gr_r != gr).sum()),
          "blocs_zero": float((xsf == 0).double().mean()),
          "err_iso": _err(deq, xs_div),                  # bruit de quantification seul
          "err_cum": _err(deq, xg_ref),
          "err_xs_vs_exact": _err(xs_div, xg_ref)}       # division bf16 du noyau
    if distinct:
        a1 = qa[1][0]
        xq2_r, xsf2_r, gr2_r = quant_act_ref(xs2_in, a1[1], a1[2])
        deq2 = _deq_act(xq2, xsf2, gr2).double()
        e1.update({"up_octets_diff_ref": int((xq2_r != xq2).sum() + (xsf2_r != xsf2).sum() + (gr2_r != gr2).sum()),
                   "up_blocs_zero": float((xsf2 == 0).double().mean()),
                   "up_err_iso": _err(deq2, div_bf16(xs2_in, a1[1], a1[2])), "up_err_cum": _err(deq2, xu_ref)})
    else:
        deq2 = deq
    E["1_quant_act"] = e1

    # 2. GEMM gate / up (sortie bf16 [G, Mp])
    m_i = pg[5]
    g_iso = gemm_par_expert(deq, Wg, m_i); u_iso = gemm_par_expert(deq2, Wu, m_i)
    g_cum = gemm_par_expert(xg_ref, Wg, m_i); u_cum = gemm_par_expert(xu_ref, Wu, m_i)
    E["2_gemm_gate_up"] = {"err_iso": _err(g[:, :m_i], g_iso), "err_cum": _err(g[:, :m_i], g_cum),
                           "up_err_iso": _err(u[:, :m_i], u_iso), "up_err_cum": _err(u[:, :m_i], u_cum)}

    # 3. moe_act : silu(g)·u (la division par s_d vit dans quant_act, étape 4)
    def acte(gg, uu, diviser):
        gg = gg[:, :m_i]; uu = uu[:, :m_i]
        a = torch.nn.functional.gelu(gg, approximate="tanh") if mod.act == "gelu_tanh" else torch.nn.functional.silu(gg)
        v = a * uu
        out = torch.zeros(G, I, device=dev, dtype=torch.float64)
        out[:, :m_i] = v / sd_r[:, :m_i] if diviser else v
        return out
    act_iso = acte(g.double(), u.double(), False); act_cum = acte(g_cum, u_cum, True)
    E["3_moe_act"] = {"err_iso": _err(act, act_iso), "err_cum": _err(act, acte(g_cum, u_cum, False))}

    # 4. quantification de l'entrée de down (division act/s_d[e] fusionnée)
    a2 = qa[-1][0]
    aq_r, asf_r, gra_r = quant_act_ref(act_in, a2[1], a2[2])
    deqa = _deq_act(aq, asf, gra).double()
    act_div = div_bf16(act_in, a2[1], a2[2])
    E["4_quant_act_down"] = {"octets_diff_ref": int((aq_r != aq).sum() + (asf_r != asf).sum() + (gra_r != gra).sum()),
                             "blocs_zero": float((asf == 0).double().mean()),
                             "blocs_zero_ref_nonnuls": int(((asf == 0) & (act_in.view(G, I // 16, 16).abs().amax(-1) > 0)).sum()),
                             "err_iso": _err(deqa, act_div), "err_cum": _err(deqa, act_cum)}

    # 5. GEMM down
    d_iso = gemm_par_expert(deqa, Wd, Mo); d_cum = gemm_par_expert(act_cum, Wd, Mo)
    E["5_gemm_down"] = {"err_iso": _err(d[:, :Mo], d_iso), "err_cum": _err(d[:, :Mo], d_cum)}

    # 6. réduction pondérée
    def reduit(dd):
        out = torch.zeros(t, Mo, device=dev, dtype=torch.float64)
        out.index_add_(0, tok, dd[:, :Mo] * tw.double().unsqueeze(1))
        return out
    E["6_reduce"] = {"err_iso": _err(y[:, :Mo], reduit(d.double())), "err_cum": _err(y[:, :Mo], reduit(d_cum))}

    del loaded
    torch.cuda.empty_cache()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jetons", type=int, default=16)
    ap.add_argument("--convertis", nargs="*", default=list(CONVERTIS))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("carte requise (outils/carte.sh)")
    SORTIE.mkdir(parents=True, exist_ok=True)
    x = _x_reel(args.jetons)
    out = {}
    for nom in args.convertis:
        out[nom] = sonder(nom, CONVERTIS[nom], x)
        (SORTIE / f"{nom}.json").write_text(json.dumps(out[nom], indent=1, ensure_ascii=False))
    if len(out) == 2:
        a, b = out["alpha-commun"]["etapes"], out["sans-awq"]["etapes"]
        print(f"{'étape':20s} {'iso alpha':>10s} {'iso sans':>10s} {'ratio':>7s} | {'cum alpha':>10s} {'cum sans':>10s} {'ratio':>7s}")
        coupables = []
        for et in a:
            ri = a[et]["err_iso"] / max(b[et]["err_iso"], 1e-30)
            rc = a[et]["err_cum"] / max(b[et]["err_cum"], 1e-30)
            print(f"{et:20s} {a[et]['err_iso']:10.3e} {b[et]['err_iso']:10.3e} {ri:7.2f} | {a[et]['err_cum']:10.3e} {b[et]['err_cum']:10.3e} {rc:7.2f}")
            if ri >= 3:
                coupables.append(et)
        for et in ("1_quant_act", "4_quant_act_down"):
            print(f"{et} octets≠ref alpha/sans : {a[et]['octets_diff_ref']}/{b[et]['octets_diff_ref']} ; "
                  f"blocs zéro : {a[et]['blocs_zero']:.4%}/{b[et]['blocs_zero']:.4%}")
        print("VERDICT :", (f"une étape, {coupables[0]}" if len(coupables) == 1
                            else f"{len(coupables)} étapes ≥ 3× : {coupables} (scellé : exactement une)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
