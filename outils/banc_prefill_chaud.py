#!/usr/bin/env python3
"""Banc de prefill MoE en régime chaud — celui des chiffres de
revue/mma-fp4-native-sm120.md (16 938 j/s à L=2048, bras m64e4).

    outils/carte.sh .venv/bin/python3 outils/banc_prefill_chaud.py m64e4 2048

Dénominateur : L / durée d'un generate(max_tokens=1) complet, synchronisé
aux deux bouts, moteur chaud (2 passes de chauffe), 7 répétitions, médian ± σ,
enable_prefix_cache=False ET invite DIFFÉRENTE à chaque répétition — sinon
le cache de préfixe rend un pas de ~27 ms quelle que soit L (piège du 12/09,
revue/banc-prefill-moe-12-09.md). Les compteurs prouvent le chemin pris.

Bras : b (GEMM directe wmma) | c (déquant + _grouped_mm) | a (GEMV plein) |
d (tranches de 32 par le runner) | m<bt>[e<etages>] (MMA FP4 native).
"""
import os, sys, math, time, json
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
bras = sys.argv[1]
L = int(sys.argv[2])
REP = 7
if bras == "a":
    os.environ["ACVRAM_MOE_GROUPED_MAX"] = "99999"
elif bras == "c":
    os.environ["ACVRAM_PREFILL_DEQUANT"] = "1"
elif bras == "d":
    os.environ["ACVRAM_BUDGET_JETONS"] = "32"
elif bras.startswith("m"):
    os.environ["ACVRAM_MOE_MMA"] = "1"
    reste = bras[1:]
    if "k" in reste:
        reste, ks = reste.split("k"); os.environ["ACVRAM_MOE_MMA_KS"] = ks
    if "e" in reste:
        bt, et = reste.split("e"); os.environ["ACVRAM_MOE_MMA_ETAGES"] = et
    else:
        bt = reste
    if bt: os.environ["ACVRAM_MOE_MMA_BT"] = bt
import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from regime import exiger_regime_nominal
import importlib.util
_s = importlib.util.spec_from_file_location("rm", os.path.join(_REPO, "outils/racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, "Qwen3-Coder-30B-A3B-nvfp4")
SP = SamplingParams(temperature=0.0, max_tokens=1)
loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0")
eng = Engine(loaded, None, max_batch_size=1, max_model_len=4096,
             enable_cuda_graphs=False, enable_prefix_cache=False)
c = {"fpg": 0, "fg": 0, "gemm": 0, "pile": 0, "mma": 0}
def wrap(mod, nom, cle):
    fn = getattr(mod, nom)
    def w(*a, **kw):
        c[cle] += 1
        return fn(*a, **kw)
    setattr(mod, nom, w)
for _, mod in loaded.model.named_modules():
    if type(mod).__name__ == "MoEBlock":
        wrap(mod, "_forward_prefill_grouped", "fpg"); wrap(mod, "_forward_grouped", "fg")
        wrap(mod, "_gemm", "gemm"); wrap(mod, "_pile_bf16", "pile")
        if hasattr(mod, "_gemm_mma"): wrap(mod, "_gemm_mma", "mma")
def un(rep):
    prompt = [(1000 + rep * 7919 + i * 13) % 150000 + 10 for i in range(L)]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    list(eng.generate(prompt, SP))
    torch.cuda.synchronize()
    return time.perf_counter() - t0
for r in range(2): un(100 + r)
exiger_regime_nominal(eng, autoriser_piles_inconnues=False)
for k in c: c[k] = 0
d = [un(r) for r in range(REP)]
jps = sorted(L / x for x in d)
med = jps[REP // 2]; moy = sum(jps) / REP
sig = math.sqrt(sum((x - moy) ** 2 for x in jps) / REP)
print("RESULTAT " + json.dumps({"bras": bras, "L": L, "med_jps": round(med), "sigma": round(sig),
      "ms": round(1000 * L / med, 1), "compteurs": c}))
