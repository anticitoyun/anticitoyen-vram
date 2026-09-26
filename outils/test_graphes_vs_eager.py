#!/usr/bin/env python3
"""Jetons générés sous graphes CUDA contre eager, même scénario (12 séquences,
fins à des pas différents, une arrivée en cours de lot), et contre une
version antérieure de graphs.py si GRAPHS_AVANT=<fichier> est posé.

    outils/carte.sh .venv/bin/python3 outils/test_graphes_vs_eager.py graphes
    outils/carte.sh .venv/bin/python3 outils/test_graphes_vs_eager.py eager
    GRAPHS_AVANT=/chemin/graphs_avant.py … graphes

Imprime une empreinte par séquence et la liste des divergences par rapport
au fichier JETONS_REF (json) s'il est donné.
"""
import os, sys, json, hashlib, importlib.util
_ICI = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)
bras = sys.argv[1]
avant = os.environ.get("GRAPHS_AVANT")
if avant:
    # substitue le module graphs par la version antérieure AVANT tout import du moteur
    import acvram.engine  # noqa: F401
    spec = importlib.util.spec_from_file_location("acvram.engine.graphs", avant)
    mod = importlib.util.module_from_spec(spec); sys.modules["acvram.engine.graphs"] = mod
    spec.loader.exec_module(mod)
import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))
loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0")
eng = Engine(loaded, None, max_batch_size=12, max_model_len=1024,
             enable_cuda_graphs=(bras == "graphes"), enable_prefix_cache=False)
eng._eos = set()
maxs = [60, 200, 90, 200, 120, 160, 200, 45, 200, 75, 200, 200]
sorties = {}
def ajouter(i, rep):
    sorties[i] = []
    eng.add_request([(1000 + rep * 7919 + i * 101 + j * 13) % 150000 + 10 for j in range(64)],
                    SamplingParams(temperature=0.0, max_tokens=maxs[i]), request_id=f"s{i}")
for i in range(11): ajouter(i, 0)
pas = 0
while eng.running or eng.waiting:
    for o in eng.step():
        sorties[int(o.request_id[1:])].extend(o.token_ids)
    pas += 1
    if pas == 30: ajouter(11, 0)                    # arrivée en cours de lot
emp = {f"s{i}": hashlib.sha256(repr(sorties[i]).encode()).hexdigest()[:10] for i in range(12)}
print(f"[{bras}{' AVANT' if avant else ''}] pas={pas} " + " ".join(f"{k}={v}" for k, v in emp.items()))
ref = os.environ.get("JETONS_REF")
if ref:
    r = json.load(open(ref))
    for i in range(12):
        a, b = sorties[i], r[f"s{i}"]
        d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
        if d is not None or len(a) != len(b):
            print(f"  DIVERGENCE s{i} au jeton {d} : ici={a[d] if d is not None else None} ref={b[d] if d is not None else None} (long {len(a)} vs {len(b)})")
sauve = os.environ.get("JETONS_SAUVE")
if sauve:
    json.dump({f"s{i}": sorties[i] for i in range(12)}, open(sauve, "w"))
