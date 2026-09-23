#!/usr/bin/env python3
"""Pièce 58 (Gaelle, 23/09) — l'arithmétique de la boucle du GEMM étroit, noyau seul, mêmes formes,
L2 froid (harnais de banc-etroites-latence.py) :
  (a) W8A16 « échelles hors boucle » : les échelles et zéros de la tranche sont chargés UNE fois
      ([BN, groupes]) avant la boucle sur K et appliqués depuis les registres à l'accumulateur
      fp32 par groupe — même ordre d'opérations que le servi : attendu AU BIT ;
  (b) W8A8 FP8 e4m3 natif : x quantifié par jeton (échelle fp32 par ligne), poids e4m3 par
      canal (échelle par colonne), `tl.dot` e4m3 × e4m3 → fp32, échelles en épilogue — le chemin
      de TRT-LLM ; arithmétique différente par construction : on publie l'écart relatif contre
      la référence fp32 (x_deq · w_deq), la qualité se juge ensuite par KL (porte Coder ≤ 0,74).
Sortie : µs froid/chaud, To/s (octets des poids + échelles), au bit / écart, pour le servi, (a), (b).

    outils/carte.sh python outils/gpu/mesure/banc-etroites-arithmetique.py --forme qkv --json qkv.json
"""
import argparse, importlib.util, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
_lat = importlib.util.spec_from_file_location("lat", os.path.join(os.path.dirname(os.path.abspath(__file__)), "banc-etroites-latence.py"))
lat = importlib.util.module_from_spec(_lat); _lat.loader.exec_module(lat)
occ, GE = lat.occ, lat.GE
import triton
import triton.language as tl

GPT_MAX = 16     # groupes par tranche portés en registres par (a) : 16 × 128 = 2 048 colonnes de K


@triton.jit
def _reduit_a(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, cnt_ptr, out_ptr, M, N, K, NG, gpt, tranches,
              stride_xm, stride_qn, stride_sn, stride_ys, stride_ym, stride_om,
              BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr, GPT_: tl.constexpr):
    pn = tl.program_id(0); ps = tl.program_id(1)
    rows = tl.arange(0, BM_); cols = pn * BN_ + tl.arange(0, BN_)
    masque_m = rows < M; masque_n = cols < N
    masque = masque_m[:, None] & masque_n[None, :]
    g0 = ps * gpt
    gs = g0 + tl.arange(0, GPT_)                                   # groupes de la tranche
    masque_gs = (gs < NG) & (gs < g0 + gpt)
    S = tl.load(s_ptr + cols[:, None] * stride_sn + gs[None, :], mask=masque_n[:, None] & masque_gs[None, :], other=0.0).to(tl.float32)  # [BN, GPT]
    Z = tl.load(z_ptr + cols[:, None] * stride_sn + gs[None, :], mask=masque_n[:, None] & masque_gs[None, :], other=0).to(tl.float32)
    kk = tl.arange(0, G)
    acc = tl.zeros((BM_, BN_), dtype=tl.float32)
    for i in tl.static_range(GPT_):
        g = g0 + i
        masque_g = (g < NG) & (g < g0 + gpt)
        ks = g * G + kk
        masque_k = (ks < K) & masque_g
        x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=masque_m[:, None] & masque_k[None, :], other=0.0)
        q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :], mask=masque_n[:, None] & masque_k[None, :], other=0)
        prod = tl.dot(x, tl.trans(q.to(x.dtype)))
        sx = tl.sum(x.to(tl.float32), 1)
        s_g = tl.sum(tl.where(tl.arange(0, GPT_)[None, :] == i, S, 0.0), 1)     # colonne i de S : [BN]
        z_g = tl.sum(tl.where(tl.arange(0, GPT_)[None, :] == i, Z, 0.0), 1)
        acc += (prod - sx[:, None] * z_g[None, :]) * s_g[None, :]
    tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
    tl.debug_barrier()
    n = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if n == tranches - 1:
        somme = tl.zeros((BM_, BN_), dtype=tl.float32)
        for t in range(0, tranches):
            somme += tl.load(y_ptr + t * stride_ys + rows[:, None] * stride_ym + cols[None, :], mask=masque, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :], somme.to(out_ptr.dtype.element_ty), mask=masque)
        tl.store(cnt_ptr + pn, 0)


@triton.jit
def _fp8_kernel(x_ptr, sx_ptr, w_ptr, sw_ptr, y_ptr, cnt_ptr, out_ptr, M, N, K, ktiles_par_tranche, tranches,
                stride_xm, stride_wn, stride_ys, stride_ym, stride_om,
                BM_: tl.constexpr, BN_: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0); ps = tl.program_id(1)
    rows = tl.arange(0, BM_); cols = pn * BN_ + tl.arange(0, BN_); kk = tl.arange(0, BK)
    masque_m = rows < M; masque_n = cols < N
    masque = masque_m[:, None] & masque_n[None, :]
    acc = tl.zeros((BM_, BN_), dtype=tl.float32)
    for t in range(ps * ktiles_par_tranche, (ps + 1) * ktiles_par_tranche):
        ks = t * BK + kk
        masque_k = ks < K
        x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=masque_m[:, None] & masque_k[None, :], other=0.0)
        w = tl.load(w_ptr + cols[:, None] * stride_wn + ks[None, :], mask=masque_n[:, None] & masque_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))                                   # e4m3 × e4m3 → fp32
    sx = tl.load(sx_ptr + rows, mask=masque_m, other=0.0)
    sw = tl.load(sw_ptr + cols, mask=masque_n, other=0.0)
    acc = acc * sx[:, None] * sw[None, :]
    tl.store(y_ptr + ps * stride_ys + rows[:, None] * stride_ym + cols[None, :], acc, mask=masque)
    tl.debug_barrier()
    n = tl.atomic_add(cnt_ptr + pn, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    if n == tranches - 1:
        somme = tl.zeros((BM_, BN_), dtype=tl.float32)
        for tt in range(0, tranches):
            somme += tl.load(y_ptr + tt * stride_ys + rows[:, None] * stride_ym + cols[None, :], mask=masque, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :], somme.to(out_ptr.dtype.element_ty), mask=masque)
        tl.store(cnt_ptr + pn, 0)


def gemm_a(x, t):
    M, K = x.shape; N, k_pad = t.qweight.shape; G = t.group_size; ng = k_pad // G
    tuiles_n = -(-N // GE.BN)
    tranches, gpt = GE.decouper_k(ng, tuiles_n, x.device)
    assert gpt <= GPT_MAX, (gpt, GPT_MAX)
    y = torch.empty(tranches, M, N, dtype=torch.float32, device=x.device)
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _reduit_a[(tuiles_n, tranches)](x, t.qweight, t.scales, t.zeros, y, GE._compteur(tuiles_n, x.device), out, M, N, K, ng, gpt, tranches,
                                    x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1), out.stride(0),
                                    BM_=GE.BM, BN_=GE.BN, G=G, GPT_=GPT_MAX, num_warps=4, num_stages=3)
    return out


class PoidsFP8:
    """Poids e4m3 par canal (échelle par colonne) tirés du même int8 déquantifié ; x e4m3 par jeton."""
    def __init__(self, t, x):
        w = occ.dequant_bf16(t) if hasattr(occ, "dequant_bf16") else _dequant(t)          # [N, K] bf16
        amax = w.float().abs().amax(1).clamp_min(1e-8)
        self.sw = (amax / 448.0).contiguous()
        self.w = (w.float() / self.sw[:, None]).to(torch.float8_e4m3fn).contiguous()
        ax = x.float().abs().amax(1).clamp_min(1e-8)
        self.sx = (ax / 448.0).contiguous()
        self.x = (x.float() / self.sx[:, None]).to(torch.float8_e4m3fn).contiguous()
        self.ref = (x.float() @ w.float().t())                                              # référence fp32 sur les MÊMES poids déquantifiés


def _dequant(t):
    N, K = t.qweight.shape; G = t.group_size
    q = t.qweight.float().view(N, K // G, G)
    return ((q - t.zeros.float()[:, :, None]) * t.scales.float()[:, :, None]).view(N, K).to(torch.bfloat16)


def gemm_fp8(p, M, BK=128):
    N, K = p.w.shape
    tuiles_n = -(-N // GE.BN)
    ktiles = K // BK
    tranches, kpt = GE.decouper_k(ktiles, tuiles_n, p.w.device)
    y = torch.empty(tranches, M, N, dtype=torch.float32, device=p.w.device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=p.w.device)
    _fp8_kernel[(tuiles_n, tranches)](p.x, p.sx, p.w, p.sw, y, GE._compteur(tuiles_n, p.w.device), out, M, N, K, kpt, tranches,
                                      p.x.stride(0), p.w.stride(0), y.stride(0), y.stride(1), out.stride(0),
                                      BM_=GE.BM, BN_=GE.BN, BK=BK, num_warps=4, num_stages=3)
    return out


def mesurer(nom, rep):
    dev = torch.device("cuda", torch.cuda.current_device())
    n, k = occ.FORMES_POIDS[nom]
    copies = [occ.poids_int8(n, k, graine=(hash(nom) + i) % 1000, device=dev) for i in range(lat.COPIES)]
    B = getattr(occ, "B", 12)
    x = (torch.randn(B, k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
    t0 = copies[0]
    octets_i8 = n * k + t0.scales.numel() * 2 + t0.zeros.numel()
    GE.regler_forme(None)
    ref = GE.gemm_etroit(x, t0, compact=True).clone()
    r = {"forme": nom, "n_k": [n, k], "b": B, "copies": lat.COPIES, "rep": rep, "bras": {}}

    def ligne(fns_froid, fn_chaud, octets):
        tc = occ.chrono(fn_chaud, rep); tf = lat.chrono_froid(fns_froid, max(20, rep // 10))
        mf = tf[len(tf) // 2]
        return {"us_chaud": round(tc[len(tc) // 2], 2), "us_froid": round(mf, 2), "to_s_froid": round(octets / (mf * 1e-6) / 1e12, 3), "octets": octets}
    ref32 = x.float() @ _dequant(t0).float().t()                 # référence fp32 sur les poids déquantifiés du servi
    def rms_rel(y):                                              # ||y − ref|| / ||ref|| : insensible aux passages par zéro (l écart relatif par élément y explose)
        return float((y.float() - ref32).norm() / ref32.norm())
    r["bras"]["servi W8A16"] = {**ligne([(lambda t=t: GE.gemm_etroit(x, t, compact=True)) for t in copies], lambda: GE.gemm_etroit(x, t0, compact=True), octets_i8), "au_bit": True,
                                "ecart_rms_rel": rms_rel(ref)}
    try:
        ya = gemm_a(x, t0)
        r["bras"]["(a) W8A16 échelles hors boucle"] = {**ligne([(lambda t=t: gemm_a(x, t)) for t in copies], lambda: gemm_a(x, t0), octets_i8),
                                                        "au_bit": bool(torch.equal(ya, ref)), "ecart_max": float((ya.float() - ref.float()).abs().max()),
                                                        "ecart_ulp_bf16_max": float(((ya.float() - ref.float()).abs() / (ref.float().abs().clamp_min(1e-30) * 2.0 ** -7)).max())}
    except Exception as exc:
        r["bras"]["(a) W8A16 échelles hors boucle"] = {"erreur": f"{type(exc).__name__}: {exc}"[:160]}
    try:
        ps = [PoidsFP8(t, x) for t in copies]
        p0 = ps[0]
        yb = gemm_fp8(p0, B)
        octets_fp8 = n * k + n * 4
        err = ((yb.float() - p0.ref).abs() / p0.ref.abs().clamp_min(1e-3))
        r["bras"]["(b) W8A8 FP8 e4m3"] = {**ligne([(lambda p=p: gemm_fp8(p, B)) for p in ps], lambda: gemm_fp8(p0, B), octets_fp8),
                                          "au_bit": False, "ecart_rms_rel": rms_rel(yb), "ecart_rel_median": float(err.median()), "ecart_rel_p99": float(err.flatten().kthvalue(int(err.numel() * 0.99)).values),
                                          "ecart_rel_max": float(err.max())}
    except Exception as exc:
        r["bras"]["(b) W8A8 FP8 e4m3"] = {"erreur": f"{type(exc).__name__}: {exc}"[:160]}
    base = r["bras"]["servi W8A16"]["us_froid"]
    r["verdict"] = {k: {"gain_pct": round(100 * (base - v["us_froid"]) / base, 1), "to_s": v["to_s_froid"]} for k, v in r["bras"].items() if "us_froid" in v}
    print("RESULTAT " + json.dumps({"forme": nom, **r["verdict"], "erreurs": [k for k, v in r["bras"].items() if "erreur" in v]}, ensure_ascii=False), flush=True)
    for k, v in r["bras"].items():
        print(json.dumps({"bras": k, **v}, ensure_ascii=False), flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forme", choices=list(occ.FORMES_POIDS), required=True)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = mesurer(a.forme, a.rep)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
