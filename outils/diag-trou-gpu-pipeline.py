#!/usr/bin/env python3
"""Controle (1) de chef, 14/09 soir, etendu : (a) trou GPU reel entre
rejouer_suivant(n) et rejouer_suivant(n+1) sous ACVRAM_PIPELINE=1, (b) meme
mesure de temps MUR par pas (host, time.perf_counter) sous PIPELINE=0 sur la
MEME fenetre de contexte (memes 30 pas, meme invite) -- pour comparer sans le
biais de croissance du contexte sur 200 pas (l'attention coute plus cher a
mesure que le lot grandit, meme SANS aucun changement de recouvrement).

    outils/carte.sh .venv/bin/python outils/diag-trou-gpu-pipeline.py
"""
import os
import time

import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
N_SLOTS = 12
N_PAS = 30


def invite(k, n=256):
    return [(k * 104729 + i * 7919) % 150000 + 10 for i in range(n)]


def mesurer_trou(pipeline: bool):
    os.environ["ACVRAM_PIPELINE"] = "1" if pipeline else "0"
    os.environ["ACVRAM_REPIN"] = "0"
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=2048)
    engine = Engine(loaded, None, max_batch_size=N_SLOTS, max_model_len=2048)
    engine.pipeline_actif = pipeline
    engine.warm_graphs()

    evenements = []
    orig = engine.graphs.rejouer_suivant

    def espion():
        debut = torch.cuda.Event(enable_timing=True)
        debut.record()
        out = orig()
        fin = torch.cuda.Event(enable_timing=True)
        fin.record()
        evenements.append((debut, fin))
        return out

    engine.graphs.rejouer_suivant = espion

    params = SamplingParams(temperature=0.0, max_tokens=200)
    for k in range(N_SLOTS):
        engine.add_request(invite(k), params, request_id=f"d{k}")

    t0 = time.perf_counter()
    for _ in range(N_PAS):
        engine.step()
    torch.cuda.synchronize()
    mur_total_ms = (time.perf_counter() - t0) * 1000

    trous = [evenements[i - 1][1].elapsed_time(evenements[i][0])
             for i in range(1, len(evenements))]
    durees = [d.elapsed_time(f) for d, f in evenements]

    del engine, loaded
    torch.cuda.empty_cache()
    return mur_total_ms, trous, durees


for pipeline in (False, True):
    mur_total_ms, trous, durees = mesurer_trou(pipeline)
    label = "PIPELINE=1" if pipeline else "PIPELINE=0"
    print(f"\n[{label}]", flush=True)
    print(f"  mur/pas moyen (host, {N_PAS} pas, un seul sync final) = "
          f"{mur_total_ms/N_PAS:.3f} ms", flush=True)
    print(f"  rejeu (GPU, pas 2..{N_PAS}) moyenne={sum(durees[1:])/len(durees[1:]):.3f} ms  "
          f"5 derniers={[round(x,2) for x in durees[-5:]]}", flush=True)
    print(f"  trou avant rejeu suivant, moyenne={sum(trous)/len(trous):.4f} ms  "
          f"5 derniers={[round(x,3) for x in trous[-5:]]}", flush=True)
