#!/usr/bin/env python3
"""Bead runner (14/09) : jetons émis bit-identiques entre ACVRAM_PIPELINE=0
et =1, sur un scénario qui exerce fins de séquence à des pas différents ET
une arrivée en cours de lot (bead pds, condition posée par Jérôme).

Deux moteurs, même modèle chargé une fois, même script d'appels rejoué à
l'identique sur les deux (add_request/step dans le même ordre).

État 14/09 soir : en attente d'un correctif dans `graphs.py` --
`diag-graphes-vs-eager.py` a isolé un bogue de fond dans `preparer`/`_fill`,
présent MÊME avec ACVRAM_PIPELINE jamais activé -- ce test hérite donc de
ses divergences tant que ce correctif n'est pas fait.
"""
import os, sys
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ICI))
os.environ["ACVRAM_REPIN"] = "0"          # pas de swap pendant le test

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
    """Rejoue EXACTEMENT le même script sur `engine` : 11 séquences dès le
    départ (max_tokens variés -> des fins à des pas différents), une 12e
    admise au pas 30 (arrivée en cours de lot -> un créneau repris)."""
    engine._eos = set()
    max_tokens = [40, 60, 80, 100, 120, 140, 160, 180, N_PAS, N_PAS, N_PAS]
    id_vers_rid: dict[int, str] = {}
    for k in range(N_SEQ_INIT):
        rid = f"s{k}"
        seq = engine.add_request(invite(k), SamplingParams(temperature=0.0, max_tokens=max_tokens[k]),
                                 request_id=rid)
        id_vers_rid[seq.id] = rid
    tokens: dict[str, list[int]] = {f"s{k}": [] for k in range(N_SEQ_INIT)}
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

engine_a = Engine(loaded, None, max_batch_size=12, max_model_len=1024)
engine_a.pipeline_actif = False
tokens_a = scenario(engine_a)
print("PIPELINE=0 termine, sequences:", {k: len(v) for k, v in tokens_a.items()}, flush=True)
del engine_a
torch.cuda.empty_cache()

engine_b = Engine(loaded, None, max_batch_size=12, max_model_len=1024)
engine_b.pipeline_actif = True
tokens_b = scenario(engine_b)
print("PIPELINE=1 termine, sequences:", {k: len(v) for k, v in tokens_b.items()}, flush=True)
del engine_b
torch.cuda.empty_cache()

assert tokens_a.keys() == tokens_b.keys(), (tokens_a.keys(), tokens_b.keys())
ok = True
for rid in tokens_a:
    if tokens_a[rid] != tokens_b[rid]:
        ok = False
        print(f"DIVERGENCE {rid} : {len(tokens_a[rid])} vs {len(tokens_b[rid])} jetons", flush=True)
        n = min(len(tokens_a[rid]), len(tokens_b[rid]))
        for i in range(n):
            if tokens_a[rid][i] != tokens_b[rid][i]:
                print(f"  premier ecart au jeton {i}: {tokens_a[rid][i]} vs {tokens_b[rid][i]}", flush=True)
                break
if ok:
    print("FAIT / TESTÉ: jetons bit-identiques PIPELINE=0 vs 1 sur "
          f"{len(tokens_a)} séquences (fins à des pas différents + une arrivée)", flush=True)
else:
    print("ÉCHEC : divergence détectée, voir ci-dessus", flush=True)
    sys.exit(1)
