#!/usr/bin/env python3
"""Banc de réfutation du port de `kernel_unified_attention` (vLLM 0.29.0) — dossier
`revue/oceane-dossier-unified-attention-23-09.md` § 7, scellé `scratchpad/oceane-unifie-23-09/scelle.md`.

Trois bras, mêmes formes (Coder : HQ 32, HKV 4, D 128, blocs de 16), L2 froid comme `banc-attn-decoupe.py` :
48 caches distincts, 48 appels rejoués en UN graphe, µs par couche = graphe / 48.

* ``vllm`` (venv vLLM) : (a) `unified_attention` tel que servi — fp8 e4m3 par tenseur, disposition LBNHC
  (K|V entrelacés par tête), 16 segments, `seq_threshold_3D` 24 ; (b) le même noyau en INT8_PER_TOKEN_HEAD sur
  NOTRE disposition (int8 [NB,16,HKV,D] + échelles fp16 [NB,16,HKV]).
* ``acvram`` (notre venv) : (c) `paged_attention(compact=True)` au défaut.

Justesse de chaque bras : erreur relative par ligne contre une référence fp64 lue du même cache (couche 0).
Le bras vLLM et le nôtre tournent dans deux processus : le compilateur Triton diffère (confondu nommé).

    outils/carte.sh <venv>/bin/python outils/gpu/mesure/banc-attn-unifie.py {vllm|acvram} SORTIE.json
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch

COUCHES, HQ, HKV, D, PAGE = 48, 32, 4, 128, 16
CELLULES = [tuple(int(x) for x in c.split(":")) for c in
            os.environ.get("ACVRAM_BANC_CELLULES", "12:768,1:768,12:320,12:1216").split(",")]
REPET = 30
SEUIL_3D = 24          # capture vLLM [1, 2, 4, 8, 16, 24] la plus proche de 128 // HKV (triton_attn.py:139-151)
SEGMENTS = 16          # triton_attn.py:54


def godet(nblk: int) -> int:
    """`bucket_blocks` recopié pour le venv vLLM (vérifié égal au nôtre dans le bras acvram)."""
    n = 8
    while n < nblk:
        n *= 2
    return n


def tables_et_lens(b, ctx, g):
    nblk = -(-ctx // PAGE)
    n = godet(nblk)
    t = torch.zeros(b, n, dtype=torch.long, device="cuda")
    t[:, :nblk] = torch.randperm(b * nblk, device="cuda", generator=g).view(b, nblk) + 1
    return t, torch.full((b,), ctx, dtype=torch.long, device="cuda"), b * nblk + 1


def reference64(kd, vd, table, lens, q, scale):
    """kd/vd : fonctions (blocs [T], positions [T]) → [T, HKV, D] float64 déquantifiés."""
    B, n_rep = q.shape[0], HQ // HKV
    ref = torch.zeros(B, HQ, D, dtype=torch.float64, device=q.device)
    for b in range(B):
        pos = torch.arange(int(lens[b]), device=q.device)
        blk = table[b, pos // PAGE]
        k, v = kd(blk, pos % PAGE), vd(blk, pos % PAGE)
        for h in range(HQ):
            p = torch.softmax((k[:, h // n_rep] @ q[b, h].double()) * scale, 0)
            ref[b, h] = p @ v[:, h // n_rep]
    return ref


def erreur(y, ref):
    e = (y.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)
    return {"err_rel_med": float(e.median()), "err_rel_max": float(e.max())}


def chrono_graphe(f):
    for _ in range(2):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPET):
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); g.replay(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a))
    return statistics.median(ts) * 1000 / COUCHES


def noyaux_us(f, n=3):
    """Somme des durées de noyaux (CUPTI) par couche, en eager : la même grandeur que les traces nsys de la p91,
    sans les trous entre nœuds que la mesure au mur du graphe compte (deux nœuds par couche chez vLLM, un chez nous)."""
    from torch.profiler import ProfilerActivity, profile
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            f()
        torch.cuda.synchronize()
    tot = sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total > 0)
    return tot / (n * COUCHES)


def bras_vllm(res):
    from vllm.v1.attention.ops import triton_unified_attention as tua
    from vllm.v1.kv_cache_interface import KVQuantMode
    import triton
    res["modules"] = {"unified_attention": tua.__file__, "triton": triton.__version__}
    print(f"vllm : {tua.__file__} · triton {triton.__version__}", flush=True)
    f8 = torch.float8_e4m3fn
    scale = 1 / math.sqrt(D)
    g = torch.Generator(device="cuda").manual_seed(23)
    segm = dict(softmax_segm_output=torch.empty(SEUIL_3D, HQ, SEGMENTS, D, dtype=torch.float32, device="cuda"),
                softmax_segm_max=torch.empty(SEUIL_3D, HQ, SEGMENTS, dtype=torch.float32, device="cuda"),
                softmax_segm_expsum=torch.empty(SEUIL_3D, HQ, SEGMENTS, dtype=torch.float32, device="cuda"))
    for b, ctx in CELLULES:
        t64, lens64, nb = tables_et_lens(b, ctx, g)
        table, lens = t64.to(torch.int32), lens64.to(torch.int32)
        cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
        q = [torch.randn(b, HQ, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(COUCHES)]
        out = [torch.empty(b, HQ, D, dtype=torch.bfloat16, device="cuda") for _ in range(COUCHES)]
        ks_t = torch.full((1,), 0.03, dtype=torch.float32, device="cuda")
        # (a) fp8 par tenseur, disposition servie LBNHC : un bloc [16, HKV, 2D], K puis V par tête
        phys = [(torch.randn(nb, PAGE, HKV, 2 * D, device="cuda", generator=g) * 4).to(f8) for _ in range(COUCHES)]
        # (b) notre disposition int8 + échelles fp16 par (jeton, tête)
        kc = [torch.randint(-127, 128, (nb, PAGE, HKV, D), dtype=torch.int8, device="cuda", generator=g)
              for _ in range(COUCHES)]
        vc = [torch.randint(-127, 128, (nb, PAGE, HKV, D), dtype=torch.int8, device="cuda", generator=g)
              for _ in range(COUCHES)]
        ks = [torch.empty(nb, PAGE, HKV, dtype=torch.float16, device="cuda").uniform_(0.01, 0.05, generator=g)
              for _ in range(COUCHES)]
        vs = [torch.empty(nb, PAGE, HKV, dtype=torch.float16, device="cuda").uniform_(0.01, 0.05, generator=g)
              for _ in range(COUCHES)]
        commun = dict(cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=ctx, softmax_scale=scale,
                      causal=True, window_size=(-1, -1), block_table=table, softcap=0, q_descale=None,
                      seq_threshold_3D=SEUIL_3D, num_par_softmax_segments=SEGMENTS, **segm)

        def pas_a():
            for i in range(COUCHES):
                tua.unified_attention(q=q[i], k=phys[i][..., :D], v=phys[i][..., D:], out=out[i],
                                      k_descale=ks_t.expand(b, HKV), v_descale=ks_t.expand(b, HKV),
                                      kv_quant_mode=KVQuantMode.FP8_PER_TENSOR, **commun)

        def pas_b():
            for i in range(COUCHES):
                tua.unified_attention(q=q[i], k=kc[i], v=vc[i], out=out[i], k_descale=None, v_descale=None,
                                      kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                                      k_scale_cache=ks[i], v_scale_cache=vs[i], **commun)

        ligne = {"b": b, "ctx": ctx, "godet": t64.shape[1]}
        pas_a()
        ref_a = reference64(lambda bl, sl: phys[0][bl, sl, :, :D].double() * 0.03,
                            lambda bl, sl: phys[0][bl, sl, :, D:].double() * 0.03, t64, lens64, q[0], scale)
        ligne["a"] = {**erreur(out[0], ref_a), "us": round(chrono_graphe(pas_a), 2), "noyaux_us": round(noyaux_us(pas_a), 2)}
        pas_b()
        ref_b = reference64(lambda bl, sl: kc[0][bl, sl].double() * ks[0][bl, sl].double()[..., None],
                            lambda bl, sl: vc[0][bl, sl].double() * vs[0][bl, sl].double()[..., None],
                            t64, lens64, q[0], scale)
        ligne["b_"] = {**erreur(out[0], ref_b), "us": round(chrono_graphe(pas_b), 2), "noyaux_us": round(noyaux_us(pas_b), 2)}
        res["cellules"].append(ligne)
        print(f"b={b:2d} ctx={ctx:5d}  (a) {ligne['a']['us']:6.2f} µs  err {ligne['a']['err_rel_max']:.2e}"
              f"   (b) {ligne['b_']['us']:6.2f} µs  err {ligne['b_']['err_rel_max']:.2e}"
              f"   noyaux (a) {ligne['a']['noyaux_us']:6.2f} (b) {ligne['b_']['noyaux_us']:6.2f}", flush=True)
        del phys, kc, vc, ks, vs, q, out
        torch.cuda.empty_cache()


def bras_acvram(res):
    racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, racine)
    from acvram.kernels import attn_paginee as ap
    from acvram.memory.kvcache import bucket_blocks
    import triton
    if not os.path.realpath(ap.__file__).startswith(os.path.realpath(racine) + os.sep):
        raise SystemExit(f"acvram importé de {ap.__file__}, pas de l'arbre {racine}")
    assert all(bucket_blocks(n) == godet(n) for n in range(1, 200)), "godet recopié ≠ bucket_blocks"
    res["modules"] = {"attn_paginee": ap.__file__, "triton": triton.__version__,
                      "REDUC_DEROULEE": ap.REDUC_DEROULEE, "WARPS_COMPACT": ap.WARPS_COMPACT}
    print(f"acvram : {ap.__file__} · triton {triton.__version__} · déroulée {ap.REDUC_DEROULEE} · "
          f"warps {ap.WARPS_COMPACT}", flush=True)
    # Pièce 96 : variantes de NOTRE noyau (tuile BN = 16 × PAGES_PAR_TUILE, warps, C) — `servi` d'abord, référence
    # de l'écart au bit ; `C4` = C et chunk du servi (calculés à PAGES_PAR_TUILE = 4) quelle que soit la tuile.
    tranches_servi = ap._tranches

    def tranches_c4(n, b, h, dev):
        ppt, ap.PAGES_PAR_TUILE = ap.PAGES_PAR_TUILE, 4
        try:
            return tranches_servi(n, b, h, dev)
        finally:
            ap.PAGES_PAR_TUILE = ppt

    w0 = ap.WARPS_COMPACT
    variantes = {"servi": (4, w0, tranches_servi), "bn16w4": (1, 4, tranches_c4), "bn16w4-Crecalc": (1, 4, tranches_servi),
                 "bn16w8": (1, w0, tranches_c4), "bn64w4": (4, 4, tranches_servi)}
    noms = os.environ.get("ACVRAM_BANC_VARIANTES", "servi").split(",")
    assert noms[0] == "servi" and all(n in variantes for n in noms), noms

    def poser(v):
        ap.PAGES_PAR_TUILE, ap.WARPS_COMPACT, ap._tranches = variantes[v]

    scale = 1 / math.sqrt(D)
    g = torch.Generator(device="cuda").manual_seed(23)
    for b, ctx in CELLULES:
        t64, lens64, nb = tables_et_lens(b, ctx, g)
        q = [torch.randn(b, HQ, D, device="cuda", generator=g).to(torch.bfloat16) for _ in range(COUCHES)]
        kc = [torch.randint(-127, 128, (nb, PAGE, HKV, D), dtype=torch.int8, device="cuda", generator=g)
              for _ in range(COUCHES)]
        vc = [torch.randint(-127, 128, (nb, PAGE, HKV, D), dtype=torch.int8, device="cuda", generator=g)
              for _ in range(COUCHES)]
        ks = [torch.empty(nb, PAGE, HKV, dtype=torch.float16, device="cuda").uniform_(0.01, 0.05, generator=g)
              for _ in range(COUCHES)]
        vs = [torch.empty(nb, PAGE, HKV, dtype=torch.float16, device="cuda").uniform_(0.01, 0.05, generator=g)
              for _ in range(COUCHES)]
        sortie = []

        def pas_c():
            sortie[:] = [ap.paged_attention(q[i], kc[i], ks[i], vc[i], vs[i], t64, lens64, HKV, scale, 0,
                                            compact=True) for i in range(COUCHES)]

        ref = reference64(lambda bl, sl: kc[0][bl, sl].double() * ks[0][bl, sl].double()[..., None],
                          lambda bl, sl: vc[0][bl, sl].double() * vs[0][bl, sl].double()[..., None],
                          t64, lens64, q[0], scale)
        ligne = {"b": b, "ctx": ctx, "godet": t64.shape[1]}
        y0 = d0 = None
        for v in noms:
            poser(v)
            pas_c()
            y = sortie[0].double()
            dv = (y - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)
            if y0 is None:
                y0, d0 = y, dv
            C, chunk = ap._tranches(t64.shape[1], b, HKV, q[0].device)
            ecart = ((y - y0).abs().amax(-1) / y0.abs().amax(-1).clamp(min=1e-6) / 2 ** -8).max().item()
            cle = "c" if v == "servi" else v
            ligne[cle] = {**erreur(sortie[0], ref), "C": C, "chunk": chunk, "bn": 16 * ap.PAGES_PAR_TUILE,
                          "warps": ap.WARPS_COMPACT, "au_bit": bool(torch.equal(y, y0)), "ecart_2m8": round(ecart, 3),
                          "part_le_1_1": round(float((dv <= 1.1 * d0).float().mean()), 4),
                          "us": round(chrono_graphe(pas_c), 2), "noyaux_us": round(noyaux_us(pas_c), 2)}
            x = ligne[cle]
            print(f"b={b:2d} ctx={ctx:5d}  {v:15s} {x['us']:6.2f} µs (noyaux {x['noyaux_us']:6.2f})  C={C:2d} "
                  f"chunk={chunk:4d} bn={x['bn']:2d} w={x['warps']}  err {x['err_rel_max']:.2e}  "
                  f"écart {x['ecart_2m8']:.3f}·2⁻⁸  ≤1,1× {x['part_le_1_1']:.4f}  au bit {x['au_bit']}", flush=True)
        poser("servi")
        res["cellules"].append(ligne)
        del kc, vc, ks, vs, q, sortie
        torch.cuda.empty_cache()


def main() -> int:
    mode, chemin = sys.argv[1], sys.argv[2]
    res = {"mode": mode, "torch": torch.__version__, "cellules": []}
    {"vllm": bras_vllm, "acvram": bras_acvram}[mode](res)
    json.dump(res, open(chemin, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
