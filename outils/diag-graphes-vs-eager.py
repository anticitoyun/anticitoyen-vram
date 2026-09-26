#!/usr/bin/env python3
"""Diagnostic (bead runner, 14/09) : graphes (preparer+rejouer_suivant+clone,
`run()` normal, sans jamais toucher au pipeline ACVRAM_PIPELINE) contre
EAGER pur (graphes désactivés), même scénario que
`diag-pipeline-bit-identique.py` (12 séquences, fins à des pas différents,
une arrivée en cours de lot).

A trouvé un bogue de fond dans `graphs.py` (preparer/_fill), 14/09 :
2 séquences sur 12 divergent de l'eager à des jetons précis (voir
acvram-memoire/revue/*graphes-preparer-divergence*), reproductible,
INDÉPENDANT du pipeline -- gardé ici pour la reprise après correctif.
"""
import os, sys
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ICI))
os.environ["ACVRAM_REPIN"] = "0"

import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
N_SEQ_INIT = 11
N_PAS = 200


def invite(k, n=128):
    return [(1000 + k * 7919 + i * 13) % 150000 + 10 for i in range(n)]


def scenario(engine):
    engine._eos = set()
    max_tokens = [40, 60, 80, 100, 120, 140, 160, 180, N_PAS, N_PAS, N_PAS]
    id_vers_rid = {}
    for k in range(N_SEQ_INIT):
        rid = f"s{k}"
        seq = engine.add_request(invite(k), SamplingParams(temperature=0.0, max_tokens=max_tokens[k]),
                                 request_id=rid)
        id_vers_rid[seq.id] = rid
    tokens = {f"s{k}": [] for k in range(N_SEQ_INIT)}
    arrivee_faite = False
    for pas in range(N_PAS + 40):
        if not arrivee_faite and pas == 30:
            seq = engine.add_request(invite(999), SamplingParams(temperature=0.0, max_tokens=50),
                                     request_id="arrivee")
            id_vers_rid[seq.id] = "arrivee"
            tokens["arrivee"] = []
            arrivee_faite = True
        for out in engine.step():
            rid = id_vers_rid[out.sequence_id]
            tokens[rid].extend(out.token_ids)
        if not engine.running and not engine.waiting:
            break
    return tokens


loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=1024)

engine_g = Engine(loaded, None, max_batch_size=12, max_model_len=1024, enable_cuda_graphs=True)
tokens_g = scenario(engine_g)
print("GRAPHES termine:", {k: len(v) for k, v in tokens_g.items()}, flush=True)
del engine_g
torch.cuda.empty_cache()

engine_e = Engine(loaded, None, max_batch_size=12, max_model_len=1024, enable_cuda_graphs=False)
tokens_e = scenario(engine_e)
print("EAGER termine:", {k: len(v) for k, v in tokens_e.items()}, flush=True)
del engine_e
torch.cuda.empty_cache()

ok = True
for rid in tokens_g:
    if tokens_g[rid] != tokens_e[rid]:
        ok = False
        n = min(len(tokens_g[rid]), len(tokens_e[rid]))
        for i in range(n):
            if tokens_g[rid][i] != tokens_e[rid][i]:
                print(f"DIVERGENCE {rid} au jeton {i}: graphes={tokens_g[rid][i]} eager={tokens_e[rid][i]}", flush=True)
                break
print("OK, graphes==eager" if ok else "ECHEC : graphs.py seul diverge de l'eager", flush=True)
