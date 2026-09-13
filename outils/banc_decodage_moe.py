#!/usr/bin/env python3
"""Décodage MoE concurrent (b séquences) : GEMV par expert (défaut) contre la
GEMM groupée MMA forcée sur le chemin décodage.

    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py gemv 12
    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py mma 12      # bt 16, étages 4
    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py profil 12   # torch.profiler d'un pas

Coder-30B, invites de 128 jetons, N jetons décodés, EOS neutralisé, 5 rép,
médian ; compteurs de chemin (_forward_grouped / _gemm_mma). Le profil
liste les 12 premiers noyaux GPU d'un pas de décodage chaud.
"""
import os, sys, time, math, json
_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)
bras = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 12
N = int(os.environ.get("BANC_JETONS", "64"))
if bras in ("mma", "profilmma"):
    os.environ["ACVRAM_MOE_MMA"] = "1"; os.environ["ACVRAM_MOE_GROUPED_MAX"] = "0"
    os.environ.setdefault("ACVRAM_MOE_MMA_BT", "16"); os.environ.setdefault("ACVRAM_MOE_MMA_ETAGES", "4")
import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
import importlib.util
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))
loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0")
eng = Engine(loaded, None, max_batch_size=B, max_model_len=1024,
             enable_cuda_graphs=os.environ.get("BANC_GRAPHES", "0") == "1", enable_prefix_cache=False)
eng._eos = set()
c = {"fg": 0, "fpg": 0, "mma": 0}
def wrap(mod, nom, cle):
    fn = getattr(mod, nom)
    def w(*a, **kw): c[cle] += 1; return fn(*a, **kw)
    setattr(mod, nom, w)
for _, mod in loaded.model.named_modules():
    if type(mod).__name__ == "MoEBlock":
        wrap(mod, "_forward_grouped", "fg"); wrap(mod, "_forward_prefill_grouped", "fpg")
        if hasattr(mod, "_gemm_mma"): wrap(mod, "_gemm_mma", "mma")
SP = SamplingParams(temperature=0.0, max_tokens=N)
def passe(rep, profiler=False):
    for b in range(B):
        eng.add_request([(1000 + rep * 7919 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    torch.cuda.synchronize()
    for k in c: c[k] = 0
    if profiler:
        from torch.profiler import profile, ProfilerActivity
        for _ in range(3): eng.step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            eng.step(); torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.self_device_time_total > 0 and not e.key.startswith("aten::")
              and not e.key.startswith("cuda") and "Graph" not in e.key]
        tot = sum(e.self_device_time_total for e in ev); ev.sort(key=lambda e: -e.self_device_time_total)
        print(f"PROFIL b={B} pas GPU={tot/1000:.2f} ms, {len(ev)} noyaux")
        for e in ev[:16]:
            print(f"  {e.self_device_time_total/1000:7.3f} ms {100*e.self_device_time_total/tot:5.1f}%  x{e.count:<4d} {e.key[:88]}")
        while eng.running: eng.step()
        torch.cuda.synchronize()
        return None
    t0 = time.perf_counter(); pas = 0
    while eng.running:
        eng.step(); pas += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0, pas
passe(0); passe(1)
if bras in ("profil", "profilmma"):
    passe(2, profiler=True); sys.exit(0)
res = sorted(passe(10 + r) for r in range(5))
dt, pas = res[len(res) // 2]
jps = [B * p / d for d, p in res]; moy = sum(jps) / len(jps)
sig = math.sqrt(sum((x - moy) ** 2 for x in jps) / len(jps))
print("RESULTAT " + json.dumps({"bras": bras, "B": B, "pas": pas, "ms_par_pas": round(1000 * dt / pas, 2),
      "jetons_par_s": round(B * pas / dt), "sigma": round(sig), "compteurs": c}))
