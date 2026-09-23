#!/usr/bin/env python3
"""Pièce 77 (revue/oceane-piece77-hors-noyaux-b12-23-09.md) : ce que coûte le décodage b = 12 hors des noyaux, sans nsys.

  temoin SORTIE.json        graphe de N noyaux Triton vides, N ∈ {1, 64, 256, 800} : pente en µs par nœud
                            (majorant de la latence entre nœuds : contient l'exécution minimale d'un noyau vide)
  nue MODELE SORTIE.json    le moteur construit comme `serve` le construit, piloté comme le client de cellule
                            (`banc-llamacpp-16-09.py decode`) : mêmes ids d'invite, chauffe, repos de 8 s, lots de
                            BANC_JETONS jetons sur une fenêtre ≥ BANC_FENETRE_S, sans HTTP ; rend t/s et ms/pas, plus
                            le compte EXACT des nœuds de chaque graphe capturé (keep_graph → cudaGraphGetNodes)
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import sys
import time

import torch

VOCAB_APPROX, INVITE = 151936, 256


def invite(k: int, n: int) -> list:
    """Celle du client de cellule (`banc-llamacpp-16-09.py:56`), au mot près."""
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


def temoin(sortie: str) -> dict:
    import triton

    @triton.jit
    def _vide(p):
        pass

    t = torch.zeros(1, device="cuda")
    res = {}
    for n in (1, 64, 256, 800):
        for _ in range(3):
            _vide[(1,)](t)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _vide[(1,)](t)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for _ in range(n):
                _vide[(1,)](t)
        g.replay(); torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        res[n] = round(statistics.median(ts), 2)
    xs, ys = list(res), list(res.values())
    mx, my = statistics.mean(xs), statistics.mean(ys)
    pente = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    r = {"us_par_rejeu": res, "pente_us_par_noeud": round(pente, 3), "ordonnee_us": round(my - pente * mx, 2)}
    json.dump(r, open(sortie, "w"), indent=1)
    return r


def compter_noeuds(graphe) -> dict:
    from cuda.bindings import runtime as rt
    g = rt.cudaGraph_t(init_value=graphe.raw_cuda_graph())
    err, _, n = rt.cudaGraphGetNodes(g, 0)
    assert err == rt.cudaError_t.cudaSuccess, err
    err, noeuds, n = rt.cudaGraphGetNodes(g, n)
    types = collections.Counter()
    for nd in noeuds:
        err, ty = rt.cudaGraphNodeGetType(nd)
        types[ty.name.replace("cudaGraphNodeType", "")] += 1
    return {"total": int(n), **dict(types)}


def nue(modele: str, sortie: str) -> dict:
    # Garder le cudaGraph_t pour le compter : sans effet sur l'exécution (instancié au premier rejeu).
    # Sous-classe, pas functools.partial : une annotation `CUDAGraph | None` du moteur exige une classe.
    class _GrapheGarde(torch.cuda.CUDAGraph):
        def __new__(cls, keep_graph=True):
            return super().__new__(cls, keep_graph=True)

        def __init__(self, keep_graph=True):
            super().__init__(keep_graph=True)
    if os.environ.get("P77_GARDER_GRAPHE", "1") == "1":           # 79 : sans le crochet, pour comparer à serve
        torch.cuda.CUDAGraph = _GrapheGarde
    sys.path.insert(0, os.getcwd())
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.speculative import NGramProposer
    from acvram.server.chat import load_tokenizer
    b = int(os.environ.get("BANC_SLOTS", "12"))
    n_jetons = int(os.environ.get("BANC_JETONS", "1024"))
    fenetre = float(os.environ.get("BANC_FENETRE_S", "20"))
    max_len = 2304
    charge = load_model(modele, dtype=torch.bfloat16, max_model_len=max_len, max_concurrent_seqs=b)
    # comme `serve` sans --speculative (défaut ngram, cli.py) : la 64 servait ainsi
    eng = Engine(charge, load_tokenizer(modele), max_batch_size=b, max_model_len=max_len, enable_prefix_cache=True,
                 speculator=NGramProposer(), enable_cuda_graphs=os.environ.get("P77_GRAPHES", "1") == "1",
                 host_kv_gib=float(os.environ.get("P77_HOST_KV_GIB", "0")))   # serve : 8,0 par défaut (cli.py --host-kv-gib)
    eng.demarrer_service(warm_max_len=int(os.environ.get("ACVRAM_WARM_GRAPHS", "2048")))
    import gc
    gc.collect(); gc.freeze(); gc.set_threshold(50000, 20, 20)
    pas_hote: list[float] = []
    vrai_step = eng.step

    def step_chrono():
        t = time.perf_counter()
        r = vrai_step()
        pas_hote.append(time.perf_counter() - t)
        return r
    eng.step = step_chrono
    compte = [0]

    def lot(prompts, n):
        p = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        for k, ids in enumerate(prompts):
            eng.add_request(ids, p, request_id=f"r{compte[0]}-{k}")
        compte[0] += 1
        recus = 0
        while eng.running or eng.waiting:
            for o in eng.step():
                recus += len(o.token_ids)
        return recus

    lot([invite(5000 + k, INVITE) for k in range(b)], 8)
    prompts = [invite(1000 + k, INVITE) for k in range(b)]
    t_c = time.perf_counter(); chauffe = 0
    while time.perf_counter() - t_c < 5.0 and chauffe < 200:
        chauffe += lot(prompts, min(64, n_jetons))
    time.sleep(8.0)                                          # le repos de 8 s du client (mesure de la puissance au repos)
    pas_hote.clear()
    n = lots = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < fenetre:
        n += lot(prompts, n_jetons); lots += 1
    duree = time.perf_counter() - t0
    noeuds = {}
    for cle, entree in (eng.graphs.graphs.items() if eng.graphs is not None else []):
        if torch.cuda.CUDAGraph is not _GrapheGarde:
            noeuds[str(cle)] = "non compté (graphe non gardé)"
            continue
        try:
            noeuds[str(cle)] = compter_noeuds(entree["graph"])
        except Exception as exc:                             # noqa: BLE001
            noeuds[str(cle)] = f"{type(exc).__name__}: {str(exc)[:80]}"
    r = {"b": b, "jetons_par_seq": n_jetons, "lots": lots, "n_jetons": n, "duree_s": round(duree, 3),
         "jetons_s": round(n / duree, 1), "ms_par_pas_equiv": round(duree / (n / b) * 1e3, 4),
         "pas_hote_n": len(pas_hote), "pas_hote_med_us": round(statistics.median(pas_hote) * 1e6, 1),
         "pas_hote_moy_us": round(statistics.mean(pas_hote) * 1e6, 1),
         "pas_longs_20ms": {"n": sum(1 for x in pas_hote if x > 0.02), "s": round(sum(x for x in pas_hote if x > 0.02), 3)},
         "pas_courts_moy_us": round(statistics.mean([x for x in pas_hote if x <= 0.02]) * 1e6, 1),
         "host_kv_gib": float(os.environ.get("P77_HOST_KV_GIB", "0")),
         "stats": {k: v for k, v in eng.stats.to_dict().items() if any(m in k for m in ("prefix", "prefixe", "prefill", "cache"))},
         "noeuds_par_graphe": noeuds, "regime": getattr(eng, "ligne_regime", lambda: None)() if callable(getattr(eng, "ligne_regime", None)) else None}
    json.dump(r, open(sortie, "w"), indent=1, ensure_ascii=False)
    return r


def main() -> int:
    if sys.argv[1] == "temoin":
        r = temoin(sys.argv[2])
    else:
        r = nue(sys.argv[2], sys.argv[3])
    print("RESULTAT " + json.dumps(r, ensure_ascii=False)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
