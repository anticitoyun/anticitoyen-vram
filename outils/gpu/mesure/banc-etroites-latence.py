#!/usr/bin/env python3
"""Pièce 57 (Gaelle, 23/09) — le GEMM étroit int8 (M=12) est-il borné par la LATENCE PAR
CONSTRUCTION (trop peu de chargements en vol par SM) ? Micro-banc sur qkv [5120, 2048] et
o [2048, 4096] : on fait varier ce qui change les octets en vol sans changer l'arithmétique —
étages de pipeline (num_stages), warps, déroulage U (U groupes de K chargés par itération),
largeur de tuile BN (octets par fil et nombre de programmes) — et on mesure µs et To/s
**à L2 froid** (24 copies des poids en rotation dans un graphe, > L2) et à L2 chaud (pièce 35).
Toute variante est comparée AU BIT au noyau servi (`gemm_etroit(compact=True)`) : l'ordre des
sommes par groupe et par tranche est le même, une différence est un défaut.

    outils/carte.sh python outils/gpu/mesure/banc-etroites-latence.py --forme qkv --json qkv.json
"""
import argparse, importlib.util, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_occ = importlib.util.spec_from_file_location("occ", os.path.join(os.path.dirname(os.path.abspath(__file__)), "banc-etroites-occupation.py"))
occ = importlib.util.module_from_spec(_occ); _occ.loader.exec_module(occ)
from acvram.kernels import gemm_etroit as GE
import triton
import triton.language as tl

COPIES = int(os.environ.get("BANC_COPIES", "24"))          # 24 × 10,5 Mo = 252 Mo > L2 : lectures HBM
GRILLE = [(bn, st, w, u) for bn in (32, 64, 128) for st in (3, 4, 6) for w in (4, 8) for u in (1, 2)]


@triton.jit
def _acc_tranche_u(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, g0, gpt, rows, cols, masque_m, masque_n,
                   stride_xm, stride_qn, stride_sn, BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr, U: tl.constexpr):
    """`_acc_tranche` du noyau servi, avec U groupes chargés par itération (déroulage statique) ;
    même ordre de somme g0, g0+1, … : au bit."""
    kk = tl.arange(0, G)
    acc = tl.zeros((BM_, BN_), dtype=tl.float32)
    for g in range(g0, g0 + gpt, U):
        for u in tl.static_range(U):
            gg = g + u
            masque_g = (gg < NG) & (gg < g0 + gpt)
            ks = gg * G + kk
            masque_k = (ks < K) & masque_g
            x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=masque_m[:, None] & masque_k[None, :], other=0.0)
            q = tl.load(q_ptr + cols[:, None] * stride_qn + ks[None, :], mask=masque_n[:, None] & masque_k[None, :], other=0)
            s = tl.load(s_ptr + cols * stride_sn + gg, mask=masque_n & masque_g, other=0.0).to(tl.float32)
            z = tl.load(z_ptr + cols * stride_sn + gg, mask=masque_n & masque_g, other=0).to(tl.float32)
            prod = tl.dot(x, tl.trans(q.to(x.dtype)))
            sx = tl.sum(x.to(tl.float32), 1)
            acc += (prod - sx[:, None] * z[None, :]) * s[None, :]
    return acc


@triton.jit
def _etroit_reduit_v(x_ptr, q_ptr, s_ptr, z_ptr, y_ptr, cnt_ptr, out_ptr, M, N, K, NG, gpt, tranches,
                     stride_xm, stride_qn, stride_sn, stride_ys, stride_ym, stride_om,
                     BM_: tl.constexpr, BN_: tl.constexpr, G: tl.constexpr, U: tl.constexpr):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rows = tl.arange(0, BM_)
    cols = pn * BN_ + tl.arange(0, BN_)
    masque_m = rows < M
    masque_n = cols < N
    masque = masque_m[:, None] & masque_n[None, :]
    acc = _acc_tranche_u(x_ptr, q_ptr, s_ptr, z_ptr, M, N, K, NG, ps * gpt, gpt, rows, cols, masque_m, masque_n,
                         stride_xm, stride_qn, stride_sn, BM_, BN_, G, U)
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


def gemm_variante(x, t, bn, stages, warps, u):
    M, K = x.shape; N, k_pad = t.qweight.shape; G = t.group_size; ng = k_pad // G
    tuiles_n = -(-N // bn)
    tranches, gpt = GE.decouper_k(ng, tuiles_n, x.device)
    y = torch.empty(tranches, M, N, dtype=torch.float32, device=x.device)
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _etroit_reduit_v[(tuiles_n, tranches)](
        x, t.qweight, t.scales, t.zeros, y, GE._compteur(tuiles_n, x.device), out, M, N, K, ng, gpt, tranches,
        x.stride(0), t.qweight.stride(0), t.scales.stride(0), y.stride(0), y.stride(1), out.stride(0),
        BM_=GE.BM, BN_=bn, G=G, U=u, num_warps=warps, num_stages=stages)
    return out, (tuiles_n, tranches)


def chrono_froid(fns, rep):
    """Un graphe qui enchaîne les `fns` (une par copie des poids) : µs par lancement, L2 froid."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns: f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / len(fns))
    return sorted(ts)


def mesurer(nom, rep):
    dev = torch.device("cuda", torch.cuda.current_device())
    n, k = occ.FORMES_POIDS[nom]
    copies = [occ.poids_int8(n, k, graine=(hash(nom) + i) % 1000, device=dev) for i in range(COPIES)]
    x = (torch.randn(getattr(occ, "B", 12), k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
    t0 = copies[0]
    octets = n * k + t0.scales.numel() * 2 + t0.zeros.numel()
    GE.regler_forme(None)
    ref = GE.gemm_etroit(x, t0, compact=True).clone()
    r = {"forme": nom, "n_k": [n, k], "octets": octets, "b": getattr(occ, "B", 12), "copies": COPIES, "rep": rep, "lignes": {}}
    # le noyau servi, chaud et froid
    tc = occ.chrono(lambda: GE.gemm_etroit(x, t0, compact=True), rep)
    tf = chrono_froid([(lambda t=t: GE.gemm_etroit(x, t, compact=True)) for t in copies], max(20, rep // 10))
    r["lignes"]["servi 4w3s BN64 U1"] = {"us_chaud": round(tc[len(tc) // 2], 2), "us_froid": round(tf[len(tf) // 2], 2),
                                          "to_s_froid": round(octets / (tf[len(tf) // 2] * 1e-6) / 1e12, 3), "au_bit": True}
    for (bn, st, w, u) in GRILLE:
        cle = f"BN{bn} s{st} w{w} U{u}"
        try:
            y, grille = gemm_variante(x, t0, bn, st, w, u)
            tc = occ.chrono(lambda: gemm_variante(x, t0, bn, st, w, u)[0], rep)
            tf = chrono_froid([(lambda t=t: gemm_variante(x, t, bn, st, w, u)[0]) for t in copies], max(20, rep // 10))
        except Exception as exc:
            r["lignes"][cle] = {"erreur": f"{type(exc).__name__}: {exc}"[:140]}; continue
        mf = tf[len(tf) // 2]
        r["lignes"][cle] = {"us_chaud": round(tc[len(tc) // 2], 2), "us_froid": round(mf, 2), "to_s_froid": round(octets / (mf * 1e-6) / 1e12, 3),
                            "programmes": grille[0] * grille[1], "au_bit": bool(torch.equal(y, ref)), "ecart_max": float((y.float() - ref.float()).abs().max())}
        print(json.dumps({"forme": nom, "variante": cle, **r["lignes"][cle]}), flush=True)
    base = r["lignes"]["servi 4w3s BN64 U1"]["us_froid"]
    ok = {k: v for k, v in r["lignes"].items() if "us_froid" in v and v.get("au_bit")}
    meilleure = min(ok.items(), key=lambda kv: kv[1]["us_froid"])
    r["verdict"] = {"base_us_froid": base, "meilleure": meilleure[0], "us_froid": meilleure[1]["us_froid"],
                    "gain_pct": round(100 * (base - meilleure[1]["us_froid"]) / base, 1), "to_s_froid": meilleure[1]["to_s_froid"],
                    "pas_au_bit": [k for k, v in r["lignes"].items() if v.get("au_bit") is False]}
    print("RESULTAT " + json.dumps(r["verdict"]), flush=True)
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
