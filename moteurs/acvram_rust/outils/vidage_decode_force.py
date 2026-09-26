"""Test décisif de la bisection du préfill : le moteur Python SERVI remplit le KV de l'invite par son CHEMIN DE
DÉCODAGE (invite réduite à son 1er jeton, puis les jetons suivants FORCÉS un par un à la place de l'échantillon),
comme le préfill Rust v1. Rend les lignes KV 0..L-1 au format du vidage. Au bit avec le Rust → le Rust reproduit
acvram et l'écart du scellé KL est interne à acvram (préfill ≠ décodage). Usage (carte.sh) :
python vidage_decode_force.py <modèle> <invites.json> <sortie-dossier>"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.server.chat import load_tokenizer, render_chat

MODELE, INVITES, SORTIE = sys.argv[1:4]
os.makedirs(SORTIE, exist_ok=True)
tok = load_tokenizer(MODELE)
loaded = load_model(MODELE, dtype=torch.bfloat16, max_model_len=2304, max_concurrent_seqs=1)
eng = Engine(loaded, tok, max_batch_size=1, max_model_len=2304, enable_prefix_cache=False, speculator=None,
             enable_cuda_graphs=True)
eng.demarrer_service(strict=False, warm_max_len=2048)
# Le pipeline recouvert reprend le jeton suivant SUR LA CARTE (échantillonneur capturé) : le forçage par le retour
# de `_sample_only` ne l'atteindrait pas. Sans pipeline, le lot suivant se bâtit sur `seq.output_ids` (hôte) ;
# le pas reste celui des graphes (`decode_fixed`).
eng.pipeline_actif = False
modele = loaded.model
force: list = []
_orig = Engine._sample_only


def forcer(self, logits, seqs, depuis_graphe=False):
    jetons, lp = _orig(self, logits, seqs, depuis_graphe)
    if force:
        jetons = jetons.clone()
        jetons.view(-1)[0] = force.pop(0)
    return jetons, lp


Engine._sample_only = forcer
for inv in json.load(open(INVITES, encoding="utf-8")):
    ids = tok.encode(render_chat(tok, inv["messages"], True))
    L = len(ids)
    force[:] = list(ids[1:])
    seq = eng.add_request(list(ids[:1]), SamplingParams(temperature=0.0, max_tokens=L + 1, ignore_eos=True), request_id=inv["nom"])
    while not seq.finished and len(seq.output_ids) < L:
        eng.step()
    obtenu = list(seq.output_ids[:L - 1])
    if obtenu != list(ids[1:]):
        i = next((j for j, (a, b) in enumerate(zip(obtenu, ids[1:])) if a != b), min(len(obtenu), L - 1))
        raise SystemExit(f"forçage non tenu ({inv['nom']}) : {len(obtenu)} jetons, 1re divergence {i}, "
                         f"obtenu {obtenu[i:i + 3]} attendu {list(ids[1:])[i:i + 3]}, pipeline={eng.pipeline_actif}")
    blocs = list(seq.blocks)
    bs = modele.caches.get(0).cfg.block_size
    ib = torch.tensor([blocs[p // bs] for p in range(L)])
    io = torch.tensor([p % bs for p in range(L)])
    kv = {}
    for i in range(len(modele.layers)):
        c = modele.caches.get(i)
        for n in ("k", "v", "k_scale", "v_scale"):
            t = getattr(c, n)
            kv[f"couche{i}.{n}"] = t[ib.to(t.device), io.to(t.device)].contiguous().cpu()
    save_file(kv, os.path.join(SORTIE, f"kv-{inv['nom']}.safetensors"))
    eng.abort(inv["nom"]) if not seq.finished else None
    print(inv["nom"], L, flush=True)
eng.fermer()
print("ok")
