#!/usr/bin/env python3
"""Pièce 99 : énergie par famille de noyaux à horloge égale, isolée, acvram (notre venv) contre vLLM (venv vLLM).
La chaîne pose -lgc 2700 pour les DEUX processus ; ici on ne touche à aucun réglage de carte.

    energie-familles-p99.py {acvram|vllm} DOSSIER SORTIE.json

Familles, mêmes octets que les bancs qui les ont chronométrées :
  moe   harnais `banc-marlin-moe-p75.py` (pile.pt de la 75 dans DOSSIER) : V `fused_marlin_moe` / A2 port 2 GEMM
        (≈ forme w13 servie depuis la 82 ter) / A3 gate-up découpées ; 20 routages × 8 copies = 160 appels par graphe ;
  proj  qkv [5120, 2048] et o [2048, 4096], M = 12, 24 copies (L2 froide) : `gemm_etroit` int8 g128 (banc de la 43)
        contre `apply_fp4_marlin_linear` nvfp4 (le noyau dense servi par vLLM) ;
  tete  [151 936, 2048] : `gemm_etroit` int8 contre `matmul` bf16 (cutlass, comme la trace vLLM).
Mesure : un graphe CUDA des appels, rejoué en boucle ≥ P99_DUREE_S (2 s) sous `Energie` (compteur NVML mJ, exact) ;
W_net = W − W_repos, repos mesuré 2 s dans le même processus, carte déjà verrouillée ; µs par appel = durée / rejeux /
appels ; mJ_net par appel = W_net × µs. Horloge médiane et bridages par bras (paire invalide hors [2 650 ; 2 700]).
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

import torch

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
M = os.path.join(R, "outils", "gpu", "mesure")
sys.path.insert(0, M)
from energie import Energie  # noqa: E402

DUREE = float(os.environ.get("P99_DUREE_S", "2.0"))
COPIES_PROJ = int(os.environ.get("P99_COPIES_PROJ", "24"))
FORMES = {"qkv": (5120, 2048), "o": (2048, 4096), "tete": (151936, 2048)}
COPIES = {"qkv": COPIES_PROJ, "o": COPIES_PROJ, "tete": 1}
B = 12
dev = torch.device("cuda", 0)


def _mod(nom: str, chemin: str):
    sp = importlib.util.spec_from_file_location(nom, chemin)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def carte() -> dict:
    q = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return {"nvidia_smi": q}


def repos() -> dict:
    torch.cuda.synchronize()
    time.sleep(0.5)
    with Energie(periode=0.1) as e:
        time.sleep(2.0)
    return {"W": round(e.moyenne, 1), "horloge_med": statistics.median(e.horloges) if e.horloges else None,
            "bridages": sorted(e.bridages), **carte()}


def graphe(appels) -> torch.cuda.CUDAGraph:
    for f in appels:
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in appels:
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in appels:
            f()
    g.replay()
    torch.cuda.synchronize()
    return g


def energie_graphe(g: torch.cuda.CUDAGraph, n_appels: int, w_repos: float) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:           # chauffe : caches, horloge stable
        g.replay()
    torch.cuda.synchronize()
    n = 0
    with Energie(periode=0.1) as e:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < DUREE:
            for _ in range(10):
                g.replay()
            n += 10
        torch.cuda.synchronize()
    dur = e.duree
    us = dur / (n * n_appels) * 1e6
    w_net = e.moyenne - w_repos
    return {"W": round(e.moyenne, 1), "W_net": round(w_net, 1), "us_appel": round(us, 3),
            "mJ_appel": round(e.joules / (n * n_appels) * 1e3, 4), "mJ_net_appel": round(w_net * us * 1e-3, 4),
            "rejeux": n, "appels": n_appels, "duree_s": round(dur, 3), "joules": round(e.joules, 2),
            "joules_trapeze": e.joules_trapeze, "horloge_med": statistics.median(e.horloges) if e.horloges else None,
            "horloge_min": min(e.horloges) if e.horloges else None, "bridages": sorted(e.bridages),
            "temp_max": max(e.temperatures) if e.temperatures else None, "invalidations": e.invalidations,
            **carte()}


def famille_moe(mode: str, dossier: str, w_repos: float) -> dict:
    p75 = _mod("p75", os.path.join(M, "banc-marlin-moe-p75.py"))

    def chrono_e(appel, copies, rts):
        appels = [(lambda p=p, rt=rt: appel(p, rt)) for rt in rts for p in copies]   # L2 froide : 20 × 8 appels
        g = graphe(appels)
        r = energie_graphe(g, len(appels), w_repos)
        del g
        return {"l2_froide": r}

    p75.chrono = chrono_e
    p75.profil = lambda *a, **k: {}
    return p75.bras_vllm(dossier) if mode == "vllm" else p75.bras_acvram(dossier)


def famille_proj(mode: str, w_repos: float) -> dict:
    g = torch.Generator(device=dev).manual_seed(99)
    out = {}
    if mode == "acvram":
        et = _mod("etroites", os.path.join(M, "banc-etroites-noyaux.py"))
    else:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (apply_fp4_marlin_linear,
                                                                                    rand_marlin_weight_nvfp4_like)
        wsp = marlin_make_workspace_new(dev)
    for nom, (n, k) in FORMES.items():
        x = (torch.randn(B, k, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        ws = [(torch.randn(n, k, device=dev, generator=g) * 0.02).to(torch.bfloat16) for _ in range(COPIES[nom])]
        if mode == "acvram":
            appels = [et.bras_triton(w, x)[0] for w in ws]
            octets, noyau = et.octets_int8(n, k), "gemm_etroit int8 g128"
        elif nom == "tete":
            appels = [(lambda w=w: torch.matmul(x, w.t())) for w in ws]
            octets, noyau = n * k * 2, "matmul bf16"
        else:
            packs = [rand_marlin_weight_nvfp4_like(w, 16) for w in ws]
            appels = [(lambda q=q, s=s, gs=gs: apply_fp4_marlin_linear(x, q, s, gs, wsp, n, k)) for _, q, s, gs in packs]
            octets, noyau = n * k // 2 + n * k // 16, "apply_fp4_marlin_linear nvfp4"
        del ws
        gph = graphe(appels)
        r = energie_graphe(gph, len(appels), w_repos)
        del gph, appels
        r.update({"octets": octets, "to_s": round(octets / (r["us_appel"] * 1e-6) / 1e12, 3), "noyau": noyau,
                  "forme": [n, k], "copies": COPIES[nom]})
        out[nom] = r
        print(f"{nom:5s} {noyau:32s} {r['us_appel']:8.2f} µs  {r['W']:6.1f} W ({r['W_net']:+.1f} net)  "
              f"{r['mJ_net_appel']:.4f} mJ net/appel  {r['to_s']:.3f} To/s  horloge {r['horloge_med']}", flush=True)
        torch.cuda.empty_cache()
    return out


def main() -> int:
    mode, dossier, sortie = sys.argv[1], sys.argv[2], sys.argv[3]
    assert mode in ("acvram", "vllm"), mode
    res = {"mode": mode, "torch": torch.__version__, "duree_s": DUREE, "cwd": os.path.relpath(os.getcwd(), os.path.expanduser("~")), "repos": repos()}
    print(f"repos {res['repos']}", flush=True)
    w0 = res["repos"]["W"]
    res["proj"] = famille_proj(mode, w0)
    res["moe"] = famille_moe(mode, dossier, w0)
    for nom, r in res["moe"].items():
        f = r["l2_froide"]
        print(f"moe   {nom:32s} {f['us_appel']:8.2f} µs/couche  {f['W']:6.1f} W ({f['W_net']:+.1f} net)  "
              f"{f['mJ_net_appel']:.4f} mJ net/couche  horloge {f['horloge_med']}", flush=True)
    if mode == "acvram":
        import acvram
        res["acvram_file"] = os.path.relpath(acvram.__file__, os.path.expanduser("~"))
        if not os.path.realpath(acvram.__file__).startswith(os.path.realpath(R) + os.sep):
            raise SystemExit(f"acvram importé de {acvram.__file__}, pas de l'arbre {R}")
    else:
        import vllm
        res["vllm_version"] = vllm.__version__
    res["repos_fin"] = repos()
    with open(sortie, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("RESULTAT " + json.dumps({k: v for k, v in res.items() if k in ("mode", "repos", "repos_fin")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
