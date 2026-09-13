#!/usr/bin/env python3
"""Profil CPU du pas de décodage sous graphes CUDA (b séquences) : où passent
les millisecondes hors rejeu.

    outils/carte.sh .venv/bin/python3 outils/profil_cpu_pas.py 12

cProfile sur N pas de décodage (ACVRAM_CHRONO_SYNC non posé : pas de
synchronisation ajoutée), Coder-30B par défaut. Rend : durée moyenne du pas,
temps GPU d'un rejeu (événements CUDA, pour le dénominateur), les 25 fonctions
les plus coûteuses (temps propre et cumulé), et le nombre de .tolist()/.item()
/synchronize par pas — chacun est une attente de l'hôte sur la carte.
"""
import os, sys, time, cProfile, pstats, io
_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 12
N_PAS = int(os.environ.get("PROFIL_PAS", "200"))
import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
import importlib.util
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))
loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0")
eng = Engine(loaded, None, max_batch_size=B, max_model_len=1024, enable_prefix_cache=False)
eng._eos = set()
SP = SamplingParams(temperature=0.0, max_tokens=N_PAS + 8)

# compteurs d'attentes hote
cpt = {"tolist": 0, "item": 0, "synchronize": 0, "cpu": 0}
_tl, _it, _sy, _cp = torch.Tensor.tolist, torch.Tensor.item, torch.cuda.synchronize, torch.Tensor.cpu
def _w(nom, fn):
    def w(*a, **k): cpt[nom] += 1; return fn(*a, **k)
    return w
torch.Tensor.tolist = _w("tolist", _tl); torch.Tensor.item = _w("item", _it)
torch.cuda.synchronize = _w("synchronize", _sy); torch.Tensor.cpu = _w("cpu", _cp)

def admettre(rep):
    for b in range(B):
        eng.add_request([(1000 + rep * 7919 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    _sy()

admettre(0)
for _ in range(20): eng.step()               # chauffe (captures faites)
_sy()
for k in cpt: cpt[k] = 0
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for _ in range(N_PAS): eng.step()
pr.disable()
_sy()
dt = (time.perf_counter() - t0) / N_PAS
print(f"PAS moyen sous profil : {dt*1000:.2f} ms  (b={B}, {N_PAS} pas) ; attentes hote par pas : "
      + ", ".join(f"{k}={v / N_PAS:.1f}" for k, v in cpt.items()), flush=True)
# temps GPU d'un rejeu seul, pour le denominateur
if eng.graphs is not None and eng.graphs.temps_replay:
    r = sorted(eng.graphs.temps_replay[-N_PAS:]); print(f"replay median (chrono moteur) : {r[len(r)//2]:.2f} ms")
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("tottime"); ps.print_stats(25)
print("=== TEMPS PROPRE (tottime) ===")
print("\n".join(l for l in s.getvalue().splitlines() if l.strip() and not l.startswith("   Ordered")))
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative"); ps.print_stats(25)
print("=== CUMULE ===")
print("\n".join(l for l in s.getvalue().splitlines()[4:] if l.strip()))
while eng.running: eng.step()
