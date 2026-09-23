"""Bisection du préfill (suite) : ce que le préfill Python SERVI passe à `PagedKVCache.write` (k, v bf16 après rope,
positions), pour la première invite, sur quelques couches — à comparer à sec au KV stocké (vidage de référence) et au
KV du préfill Rust. Usage (sous carte.sh) : python vidage_ecriture.py <modèle> <invites.json> <sortie.safetensors>"""
import json
import sys

import torch
from safetensors.torch import save_file

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.memory.kvcache import PagedKVCache
from acvram.server.chat import load_tokenizer, render_chat

MODELE, INVITES, SORTIE = sys.argv[1:4]
COUCHES = (0, 1, 2, 20, 35)
tok = load_tokenizer(MODELE)
loaded = load_model(MODELE, dtype=torch.bfloat16, max_model_len=2304, max_concurrent_seqs=1)
eng = Engine(loaded, tok, max_batch_size=1, max_model_len=2304, enable_prefix_cache=True, speculator=None,
             enable_cuda_graphs=True)
eng.demarrer_service(strict=False, warm_max_len=2048)
indice = {id(c): i for i, c in loaded.model.caches.items()}
capture: dict = {}
actif = [False]
_write = PagedKVCache.write


def espion(self, slot_mapping, k, v, positions=None):
    i = indice.get(id(self))
    if actif[0] and i in COUCHES and positions is not None and positions.numel() > 1:
        capture[f"couche{i}.k"] = k.detach().to(torch.bfloat16).contiguous().cpu().clone()
        capture[f"couche{i}.v"] = v.detach().to(torch.bfloat16).contiguous().cpu().clone()
        capture[f"couche{i}.positions"] = positions.detach().reshape(-1).cpu().clone()
    return _write(self, slot_mapping, k, v, positions)


PagedKVCache.write = espion
inv = json.load(open(INVITES, encoding="utf-8"))[0]
ids = tok.encode(render_chat(tok, inv["messages"], True))
actif[0] = True
seq = eng.add_request(list(ids), SamplingParams(temperature=0.0, max_tokens=2), request_id=inv["nom"])
while not seq.finished:
    eng.step()
actif[0] = False
save_file(capture, SORTIE)
print("ok", inv["nom"], len(ids), sorted(capture)[:6])
