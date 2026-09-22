#!/usr/bin/env python3
"""Profil torch.profiler d'UN pp2048 vLLM — 10 premiers noyaux en ms et %,
face au profil de Laurine côté acvram (MMA 38 ms, glue, int8 dense 30 ms,
flash 11 ms sur 121 ms). Consigne de Jérôme, duel A2, 14/09/2026.

vLLM execute l'inference dans un PROCESSUS SEPARE (EngineCore,
multiprocessing) : un `torch.profiler` ouvert dans le processus appelant
(comme mon premier essai) ne voit que l'attente RPC (cudaDeviceSynchronize
84 ms, AUCUN noyau CUDA) — piege trouve en le lancant. Le mecanisme
integre de vLLM (`llm.start_profile()`/`stop_profile()`, piloté par
`VLLM_TORCH_PROFILER_DIR`) fait tourner le profiler DANS l'EngineCore.

    VLLM_TORCH_PROFILER_DIR=/tmp/trace-vllm \
      /opt/ia/vLLM/.venv/bin/python outils/profil_vllm_pp2048.py <dossier_modele> [L]
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
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

REP_TRACE = os.environ.get("VLLM_TORCH_PROFILER_DIR")
if not REP_TRACE:
    print("ECHEC / CAUSE: VLLM_TORCH_PROFILER_DIR non defini")
    sys.exit(2)
Path(REP_TRACE).mkdir(parents=True, exist_ok=True)

llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
         enable_prefix_caching=False, gpu_memory_utilization=0.85,
         attention_config={"backend": "TRITON_ATTN"}, max_model_len=4096,
         profiler_config={"profiler": "torch", "torch_profiler_dir": REP_TRACE})
SP = SamplingParams(temperature=0.0, max_tokens=1)


def prompt(rep: int) -> list:
    return [(1000 + rep * 7919 + i * 13) % 150000 + 10 for i in range(L)]


# Chauffe hors profil (compilation, cache de noyaux).
for r in range(2):
    llm.generate([prompt(100 + r)], SP, use_tqdm=False)

llm.start_profile()
llm.generate([prompt(200)], SP, use_tqdm=False)
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

# Agrege la duree CUDA (evenements "X", champ "cat" contenant "kernel" ou
# "cuda", pid du flux GPU) par nom de noyau — meme esprit que
# key_averages().table(sort_by="self_cuda_time_total").
events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
par_nom: dict[str, float] = {}
for ev in events:
    if not isinstance(ev, dict):
        continue
    cat = str(ev.get("cat", "")).lower()
    if ev.get("ph") != "X" or "kernel" not in cat:
        continue
    nom = ev.get("name", "?")
    par_nom[nom] = par_nom.get(nom, 0.0) + ev.get("dur", 0) / 1000.0  # us -> ms

total_ms = sum(par_nom.values())
top = sorted(par_nom.items(), key=lambda kv: kv[1], reverse=True)[:10]
print(f"Trace : {trace_path}")
print(f"Total noyaux CUDA agrege : {total_ms:.2f} ms")
for nom, ms in top:
    print(f"  {ms:8.2f} ms  {100*ms/total_ms if total_ms else 0:5.1f} %  {nom}")

chemin = sorties() / "profil-vllm-pp2048-trace.txt"
chemin.parent.mkdir(parents=True, exist_ok=True)
chemin.write_text(f"Trace : {trace_path}\nTotal noyaux CUDA agrege : {total_ms:.2f} ms\n" +
                  "\n".join(f"{ms:8.2f} ms  {100*ms/total_ms if total_ms else 0:5.1f} %  {nom}"
                           for nom, ms in top))
print(f"\nFAIT / TESTE: {chemin} / RESTE: rien")
