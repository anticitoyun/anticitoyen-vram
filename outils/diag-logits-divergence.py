#!/usr/bin/env python3
"""Controle (2) de chef : au premier jeton divergent (s1 jeton 2, s6 jeton
10), comparer les logits des DEUX chemins (graphes, eager) pour les deux
candidats en tete -- ecart d'un ulp bf16 (bruit numerique legitime) ou ecart
franc (bogue reel) ?"""
import os, sys
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ACVRAM_REPIN"] = "0"

import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
N_SEQ_INIT = 11


def invite(k, n=128):
    return [(1000 + k * 7919 + i * 13) % 150000 + 10 for i in range(n)]


def capturer(engine, pas_cibles):
    """pas_cibles : {indice_pas: [indices_ligne_a_capturer]}. Rend
    {indice_pas: {indice_ligne: logits_top2 (valeurs, indices)}}."""
    engine._eos = set()
    max_tokens = [40, 60, 80, 100, 120, 140, 160, 180, 200, 200, 200]
    for k in range(N_SEQ_INIT):
        engine.add_request(invite(k), SamplingParams(temperature=0.0, max_tokens=max_tokens[k]),
                           request_id=f"s{k}")
    captures = {}
    pas = 0
    orig_emit = engine._emit
    def emit_espion(logits, seqs):
        nonlocal pas
        if pas in pas_cibles:
            lignes = pas_cibles[pas]
            top2 = torch.topk(logits.to(torch.float32), 2, dim=-1)
            captures[pas] = {i: (top2.values[i].tolist(), top2.indices[i].tolist())
                             for i in lignes if i < logits.shape[0]}
        pas += 1
        return orig_emit(logits, seqs)
    engine._emit = emit_espion
    for _ in range(15):
        if not engine.running and not engine.waiting:
            break
        engine.step()
    return captures


# s1 (indice de ligne 1) diverge a son jeton 2 -> 3e pas de decodage (indice
# de pas 2, 0-indexe, en supposant aucune fin avant). s6 (ligne 6) a son
# jeton 10 -> pas d'indice 10. On capture les deux lignes a ces deux pas.
CIBLES = {2: [1], 10: [6]}

loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=1024)

engine_g = Engine(loaded, None, max_batch_size=12, max_model_len=1024, enable_cuda_graphs=True)
cap_g = capturer(engine_g, CIBLES)
del engine_g
torch.cuda.empty_cache()

engine_e = Engine(loaded, None, max_batch_size=12, max_model_len=1024, enable_cuda_graphs=False)
cap_e = capturer(engine_e, CIBLES)
del engine_e
torch.cuda.empty_cache()

print("graphes:", cap_g, flush=True)
print("eager  :", cap_e, flush=True)

for pas, lignes in CIBLES.items():
    for i in lignes:
        vg, ig = cap_g[pas][i]
        ve, ie = cap_e[pas][i]
        print(f"\npas={pas} ligne={i}", flush=True)
        print(f"  graphes: top1={ig[0]} val={vg[0]:.6f}  top2={ig[1]} val={vg[1]:.6f}  ecart={vg[0]-vg[1]:.6f}", flush=True)
        print(f"  eager  : top1={ie[0]} val={ve[0]:.6f}  top2={ie[1]} val={ve[1]:.6f}  ecart={ve[0]-ve[1]:.6f}", flush=True)
        # meme candidat top1 des deux cotes ? sinon, quel est l'ecart de logit
        # du candidat de L'AUTRE chemin, mesure sur CE chemin ?
        if ig[0] != ie[0]:
            print(f"  candidats top1 DIFFERENTS : {ig[0]} (graphes) vs {ie[0]} (eager)", flush=True)
