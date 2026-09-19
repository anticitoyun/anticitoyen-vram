#!/usr/bin/env python3
"""Décodage MoE concurrent (b séquences) : GEMV par expert (défaut) contre la
GEMM groupée MMA forcée sur le chemin décodage.

    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py gemv 12
    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py mma 12      # bt 16, étages 4
    outils/carte.sh .venv/bin/python3 outils/banc_decodage_moe.py profil 12   # torch.profiler d'un pas
    ncu ... outils/banc_decodage_moe.py ncu 12   # BANC_PAS_NCU pas dans une plage NVTX "mesure"

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
from regime import exiger_regime_nominal
import importlib.util
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))
loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0", max_model_len=1024, max_concurrent_seqs=B)   # le Plan connaît le lot (garde kv_planned_seqs, 17/09)
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
        print(f"PROFIL b={B} pas GPU={tot/1000:.2f} ms, {len(ev)} noyaux, {sum(e.count for e in ev)} lancements")
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
exiger_regime_nominal(eng, autoriser_piles_inconnues=False)
if bras == "experts":
    # Compte, au pas que le bras ncu profile (prefill + 3 pas, puis le 4e), les
    # experts distincts et le max de jetons par expert de chaque couche, et les
    # octets de poids que ce routage impose (poids + échelles des 3 projections
    # par expert distinct) — le dénominateur exact des octets DRAM du MoE.
    releve = []
    def hook(mod, nom="_route"):
        fn = getattr(mod, nom)
        def w(x):
            topw, topi = fn(x)
            if releve is not None and releve_actif[0]:
                cnt = torch.bincount(topi.reshape(-1).to(torch.int64), minlength=len(mod.experts))
                releve.append((int((cnt > 0).sum()), int(cnt.max()), topi.shape[0]))
            return topw, topi
        setattr(mod, nom, w)
    releve_actif = [False]
    blocs = [m for _, m in loaded.model.named_modules() if type(m).__name__ == "MoEBlock"]
    for m in blocs: hook(m)
    for b in range(B):
        eng.add_request([(1000 + 2 * 7919 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    for _ in range(3): eng.step()
    torch.cuda.synchronize(); releve_actif[0] = True
    eng.step(); torch.cuda.synchronize(); releve_actif[0] = False
    st = blocs[0]._stacks
    E = st["gate_proj"][1].shape[0]
    par_expert = sum(st[n][1].numel() + st[n][2].numel() for n in ("gate_proj", "up_proj", "down_proj")) / E
    distincts = sum(r[0] for r in releve); cmax = max(r[1] for r in releve)
    print(f"EXPERTS pas mesure : {len(releve)} couches, t={releve[0][2]}, experts distincts total={distincts} "
          f"(moy {distincts/len(releve):.1f}/couche), max jetons/expert={cmax}, "
          f"octets poids+echelles par expert={par_expert/1e6:.3f} Mo -> {distincts*par_expert/1e9:.3f} Go de poids MoE au pas")
    sys.exit(0)
if bras == "ncu":
    # Sous ncu (--nvtx --nvtx-include "mesure/") : seuls les noyaux des
    # BANC_PAS_NCU pas de decodage b=B, apres le prefill, sont profiles.
    for b in range(B):
        eng.add_request([(1000 + 2 * 7919 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    for _ in range(3): eng.step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("mesure")
    for _ in range(int(os.environ.get("BANC_PAS_NCU", "4"))): eng.step()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print("NCU_OK", B, c); sys.exit(0)
if bras in ("profil", "profilmma"):
    passe(2, profiler=True); sys.exit(0)
res = sorted(passe(10 + r) for r in range(5))
dt, pas = res[len(res) // 2]
jps = [B * p / d for d, p in res]; moy = sum(jps) / len(jps)
sig = math.sqrt(sum((x - moy) ** 2 for x in jps) / len(jps))
print("RESULTAT " + json.dumps({"bras": bras, "B": B, "pas": pas, "ms_par_pas": round(1000 * dt / pas, 2),
      "jetons_par_s": round(B * pas / dt), "sigma": round(sig), "compteurs": c}))
