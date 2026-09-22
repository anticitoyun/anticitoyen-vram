#!/usr/bin/env python3
"""Profil torch.profiler du pas de DÉCODAGE vLLM à 12 séquences
concurrentes — vis-à-vis de notre 569 t/s (Laure). Même mécanisme que
outils/profil_vllm_pp2048.py (profiler intégré, EngineCore en processus
séparé), même invites que outils/banc_decode_vllm.py. Jérôme, 14/09/2026.

    VLLM_TORCH_PROFILER_DIR=/tmp/trace-vllm-decode \
      /opt/ia/vLLM/.venv/bin/python outils/profil_vllm_decode12.py <dossier_modele>
"""
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import sys as _s2, pathlib as _p2
_s2.path.insert(0, str(_p2.Path(__file__).resolve().parent.parent))
from outils._chemins import sorties

from vllm import LLM, SamplingParams

chemin_modele = sys.argv[1]
SLOTS = 12
CTX = 2048
VOCAB_APPROX = 150000

REP_TRACE = os.environ.get("VLLM_TORCH_PROFILER_DIR")
if not REP_TRACE:
    print("ECHEC / CAUSE: VLLM_TORCH_PROFILER_DIR non defini")
    sys.exit(2)
Path(REP_TRACE).mkdir(parents=True, exist_ok=True)


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
         enable_prefix_caching=False, gpu_memory_utilization=0.85,
         attention_config={"backend": "TRITON_ATTN"}, max_model_len=CTX,
         profiler_config={"profiler": "torch", "torch_profiler_dir": REP_TRACE})

prompts = [invite(1000 + k, min(256, CTX // 4)) for k in range(SLOTS)]

# Chauffe hors profil.
llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)

# Profil borne a quelques pas de decodage seulement (max_tokens petit) —
# on veut le PAS de decodage, pas tout le lot de 200 jetons comme le
# banc d'energie ; quelques pas suffisent pour un profil de noyaux
# representatif et bornent la taille de la trace.
llm.start_profile()
llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
            use_tqdm=False)
llm.stop_profile()

fichiers = sorted(glob.glob(os.path.join(REP_TRACE, "*.json*")),
                  key=os.path.getmtime, reverse=True)
if not fichiers:
    print(f"ECHEC / CAUSE: aucune trace ecrite dans {REP_TRACE}")
    sys.exit(2)
trace_path = fichiers[0]
ouvre = gzip.open if trace_path.endswith(".gz") else open
with ouvre(trace_path, "rt") as fh:
    trace = json.load(fh)

events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
par_nom: dict[str, float] = {}
n_appels: dict[str, int] = {}
for ev in events:
    if not isinstance(ev, dict):
        continue
    cat = str(ev.get("cat", "")).lower()
    if ev.get("ph") != "X" or "kernel" not in cat:
        continue
    nom = ev.get("name", "?")
    par_nom[nom] = par_nom.get(nom, 0.0) + ev.get("dur", 0) / 1000.0
    n_appels[nom] = n_appels.get(nom, 0) + 1

total_ms = sum(par_nom.values())
top = sorted(par_nom.items(), key=lambda kv: kv[1], reverse=True)[:10]
lignes = [f"Trace : {trace_path}",
         f"Total noyaux CUDA agrege (8 pas de decodage, 12 sequences) : {total_ms:.2f} ms"]
for nom, ms in top:
    lignes.append(f"  {ms:8.2f} ms  {100*ms/total_ms if total_ms else 0:5.1f} %  "
                  f"({n_appels[nom]:3d} appels)  {nom}")
sortie = "\n".join(lignes)
print(sortie)

chemin = sorties() / "profil-vllm-decode12-trace.txt"
chemin.parent.mkdir(parents=True, exist_ok=True)
chemin.write_text(sortie)
print(f"\nFAIT / TESTE: {chemin} / RESTE: rien")
