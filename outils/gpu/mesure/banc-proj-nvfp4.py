#!/usr/bin/env python3
"""Pièce 100 A, J1 : projections d'attention du Coder en nvfp4 par le Marlin DENSE de vLLM 0.29 (W4A16,
`apply_fp4_marlin_linear` → `ops.marlin_gemm`, repack `gptq_marlin_repack`) contre notre int8 SERVI (poids
symétrique par canal comme l'alias qkvo-i8c, appelé par le dispatcheur `acvram.kernels.backends.matmul` : la vue g128
+ `gemm_etroit` à 2 ≤ M ≤ 16, `int8_gemv`/`narrow_gemm` à M = 1 — le chemin pris est compté et imprimé).

Formes : qkv [N 5 120, K 2 048], o [N 2 048, K 4 096] ; M ∈ {12, 1}. L2 froid : 48 poids distincts (un par couche),
48 appels rejoués en UN graphe, µs par couche = mur / 48 (médiane de 30 rejeux) ; somme des noyaux CUPTI en eager en
information. Justesse de chaque bras : erreur relative max par ligne contre x · W déquantifié en fp64 (preuve que le
bras calcule ce qu'il dit, pas un critère du port).

Bras `acvram-nvfp4` (information, suite (1) du chef) : NOTRE nvfp4 par le même dispatcheur (`quantize_nvfp4`, une
échelle globale par poids) — `gemm_dense_etroit` à 4 ≤ M ≤ 32, `nvfp4_gemv` en dessous ; chemin compté.

Pièce 101, bras `acvram-marlin` : UNE disposition (poids repackés Marlin par `marlin_port.preparer_pile`, E = 1) servie
par deux de nos noyaux déjà en place — `gemm_moe` (Marlin MoE porté, E = 1, top_k = 1 : GEMM dense) et
`nvfp4_gemv_marlin` (GEMV sur disposition Marlin, E = 1, une paire par ligne). Lots par BANC_LOTS (défaut 12,1).

    outils/carte.sh <venv>/bin/python outils/gpu/mesure/banc-proj-nvfp4.py {vllm|acvram|acvram-nvfp4|acvram-marlin} SORTIE.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

COUCHES, REPET = 48, 30
FORMES = {"qkv": (5120, 2048), "o": (2048, 4096)}
LOTS = [int(v) for v in os.environ.get("BANC_LOTS", "12,1").split(",")]


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
    from torch.profiler import ProfilerActivity, profile
    f(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            f()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total > 0) / (n * COUCHES)


def erreur(y, ref):
    e = (y.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-12)
    return float(e.max())


def bras_vllm(res):
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mf
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    import triton
    import vllm
    res["modules"] = {"marlin_utils_fp4": mf.__file__, "vllm": vllm.__version__, "triton": triton.__version__}
    print(f"vllm {vllm.__version__} : {mf.__file__}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(100)
    ws = marlin_make_workspace_new(torch.device("cuda"))
    for nom, (N, K) in FORMES.items():
        poids = []
        for _ in range(COUCHES):
            w = torch.randn(N, K, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
            ref, qw, sc, gs = mf.rand_marlin_weight_nvfp4_like(w, 16)
            poids.append((ref if not poids else None, qw, sc, gs))
        for M in LOTS:
            x = [torch.randn(M, K, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(COUCHES)]
            y = [None] * COUCHES

            def pas():
                for i in range(COUCHES):
                    y[i] = mf.apply_fp4_marlin_linear(x[i], poids[i][1], poids[i][2], poids[i][3], ws, N, K)
            pas()
            err = erreur(y[0], x[0].double() @ poids[0][0].double())      # ref = Wᵀ [K, N]
            cel = {"forme": nom, "N": N, "K": K, "M": M, "us": round(chrono_graphe(pas), 2),
                   "noyaux_us": round(noyaux_us(pas), 2), "err_rel_max": err,
                   "octets_poids": int(poids[1][1].numel() * poids[1][1].element_size()
                                       + poids[1][2].numel() * poids[1][2].element_size())}
            res["cellules"].append(cel)
            print(f"{nom:3s} M={M:2d}  marlin nvfp4 {cel['us']:6.2f} µs (noyaux {cel['noyaux_us']:6.2f})  "
                  f"err {err:.2e}  poids {cel['octets_poids'] / 1e6:.2f} Mo", flush=True)
        del poids
        torch.cuda.empty_cache()


def bras_acvram(res):
    racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, racine)
    from acvram import kernels as K_
    from acvram.kernels import backends
    from acvram.quant.formats import _quantize_int8, _dequantize_int8
    import triton
    if not os.path.realpath(K_.__file__).startswith(os.path.realpath(racine) + os.sep):
        raise SystemExit(f"acvram importé de {K_.__file__}, pas de l'arbre {racine}")
    res["modules"] = {"kernels": K_.__file__, "triton": triton.__version__}
    print(f"acvram : {K_.__file__} · triton {triton.__version__}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(100)
    for nom, (N, K) in FORMES.items():
        poids = []
        for _ in range(COUCHES):
            w = torch.randn(N, K, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
            poids.append(_quantize_int8(w, group_size=K, symmetric=True))      # par canal, comme qkvo-i8c
        ref_w = _dequantize_int8(poids[0], torch.float32).double()
        for M in LOTS:
            x = [torch.randn(M, K, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(COUCHES)]
            y = [None] * COUCHES

            def pas():
                for i in range(COUCHES):
                    y[i] = backends.matmul(x[i], poids[i])
            K_.CHEMINS_INT8.clear()
            pas()
            chemins = dict(K_.CHEMINS_INT8)
            err = erreur(y[0], x[0].double() @ ref_w[:, :K].t())
            cel = {"forme": nom, "N": N, "K": K, "M": M, "chemins_int8": chemins, "us": round(chrono_graphe(pas), 2),
                   "noyaux_us": round(noyaux_us(pas), 2), "err_rel_max": err, "octets_poids": int(poids[1].nbytes)}
            res["cellules"].append(cel)
            print(f"{nom:3s} M={M:2d}  int8 servi {cel['us']:6.2f} µs (noyaux {cel['noyaux_us']:6.2f})  err {err:.2e}  "
                  f"poids {cel['octets_poids'] / 1e6:.2f} Mo  chemin {chemins}", flush=True)
        del poids
        torch.cuda.empty_cache()


def bras_acvram_nvfp4(res):
    racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, racine)
    from acvram import kernels as K_
    from acvram.kernels import backends, gemm_dense_etroit as gde
    from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4
    import triton
    if not os.path.realpath(K_.__file__).startswith(os.path.realpath(racine) + os.sep):
        raise SystemExit(f"acvram importé de {K_.__file__}, pas de l'arbre {racine}")
    appels = {"gemm_dense_etroit": 0}
    _gde = gde.gemm_dense_etroit

    def compte(*a, **k):
        appels["gemm_dense_etroit"] += 1
        return _gde(*a, **k)
    gde.gemm_dense_etroit = compte
    res["modules"] = {"kernels": K_.__file__, "triton": triton.__version__, "DENSE_NVFP4": K_._DENSE_NVFP4}
    print(f"acvram nvfp4 : {K_.__file__} · triton {triton.__version__} · DENSE_NVFP4={K_._DENSE_NVFP4}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(100)
    for nom, (N, K) in FORMES.items():
        poids = []
        for _ in range(COUCHES):
            w = torch.randn(N, K, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
            poids.append(quantize_nvfp4(w))
        ref_w = dequantize_nvfp4(poids[0], torch.float32).double()[:, :K]
        for M in LOTS:
            x = [torch.randn(M, K, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(COUCHES)]
            y = [None] * COUCHES

            def pas():
                for i in range(COUCHES):
                    y[i] = backends.matmul(x[i], poids[i])
            appels["gemm_dense_etroit"] = 0
            pas()
            chemin = "gemm_dense_etroit" if appels["gemm_dense_etroit"] == COUCHES else f"autre({appels['gemm_dense_etroit']} dense)"
            err = erreur(y[0], x[0].double() @ ref_w.t())
            cel = {"forme": nom, "N": N, "K": K, "M": M, "chemin": chemin, "us": round(chrono_graphe(pas), 2),
                   "noyaux_us": round(noyaux_us(pas), 2), "err_rel_max": err, "octets_poids": int(poids[1].nbytes)}
            res["cellules"].append(cel)
            print(f"{nom:3s} M={M:2d}  nvfp4 acvram {cel['us']:6.2f} µs (noyaux {cel['noyaux_us']:6.2f})  err {err:.2e}  "
                  f"poids {cel['octets_poids'] / 1e6:.2f} Mo  chemin {chemin}", flush=True)
        del poids
        torch.cuda.empty_cache()


def bras_acvram_marlin(res):
    racine = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, racine)
    from acvram import kernels as K_
    from acvram.kernels import marlin_port as mp
    from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4
    import triton
    if not os.path.realpath(K_.__file__).startswith(os.path.realpath(racine) + os.sep):
        raise SystemExit(f"acvram importé de {K_.__file__}, pas de l'arbre {racine}")
    ext = K_.get_extension()
    assert ext is not None and hasattr(ext, "nvfp4_gemv_marlin"), "extension sans nvfp4_gemv_marlin"
    mp.charger()
    res["modules"] = {"kernels": K_.__file__, "marlin_port": mp.__file__, "triton": triton.__version__}
    print(f"acvram marlin : {mp.__file__}", flush=True)
    dev = torch.device("cuda")
    ws = mp.espace_travail(dev)
    g = torch.Generator(device="cuda").manual_seed(100)
    for nom, (N, K) in FORMES.items():
        poids = []
        for i in range(COUCHES):
            w = torch.randn(N, K, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
            t = quantize_nvfp4(w)
            wm, sm, gm = mp.preparer_pile(t.qweight[None], t.block_scale[None], t.global_scale.reshape(1).float())
            poids.append((wm, sm, gm, dequantize_nvfp4(t, torch.float32).double()[:, :K] if i == 0 else None))
        for M in LOTS:
            x = [torch.randn(M, K, device="cuda", generator=g, dtype=torch.bfloat16) for _ in range(COUCHES)]
            ref = x[0].double() @ poids[0][3].t()
            topk = torch.zeros(M, 1, dtype=torch.int32, device=dev)
            bs = mp.choisir_block_size(M, 1, 1)
            sorted_ids, expert_ids, npost = mp.aligner_blocs(topk, bs, 1)
            tw = torch.ones(M, 1, dtype=torch.float32, device=dev)
            eid = torch.zeros(M, dtype=torch.int32, device=dev)
            tok = torch.arange(M, dtype=torch.int32, device=dev)
            y = [None] * COUCHES

            def pas_moe():
                for i in range(COUCHES):
                    y[i] = mp.gemm_moe(x[i], poids[i][0], poids[i][1], poids[i][2], sorted_ids, expert_ids, npost,
                                       tw, bs, 1, M, N, K, ws)

            def pas_dense():
                for i in range(COUCHES):
                    y[i] = mp.gemm_dense(x[i], poids[i][0][0], poids[i][1][0], poids[i][2], N, K, ws)

            def pas_gemv():
                for i in range(COUCHES):
                    y[i] = ext.nvfp4_gemv_marlin(poids[i][0], poids[i][1], poids[i][2], eid, tok, x[i], K, N, None)
            bras_liste = [("marlin_moe_E1", pas_moe), ("gemv_marlin_E1", pas_gemv)]
            if hasattr(mp.charger(), "marlin_gemm"):                   # pièce 101 : Marlin dense porté
                bras_liste.insert(0, ("marlin_dense_port", pas_dense))
            for bras, f in bras_liste:
                f()
                err = erreur(y[0].reshape(M, -1)[:, :N].float(), ref)
                cel = {"forme": nom, "N": N, "K": K, "M": M, "bras": bras, "us": round(chrono_graphe(f), 2),
                       "noyaux_us": round(noyaux_us(f), 2), "err_rel_max": err,
                       "octets_poids": int(sum(v.numel() * v.element_size() for v in poids[1][:3]))}
                res["cellules"].append(cel)
                print(f"{nom:3s} M={M:2d}  {bras:15s} {cel['us']:6.2f} µs (noyaux {cel['noyaux_us']:6.2f})  err {err:.2e}  "
                      f"poids {cel['octets_poids'] / 1e6:.2f} Mo", flush=True)
        del poids
        torch.cuda.empty_cache()


def main() -> int:
    mode, chemin = sys.argv[1], sys.argv[2]
    res = {"mode": mode, "torch": torch.__version__, "cellules": []}
    {"vllm": bras_vllm, "acvram": bras_acvram, "acvram-nvfp4": bras_acvram_nvfp4,
     "acvram-marlin": bras_acvram_marlin}[mode](res)
    json.dump(res, open(chemin, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
