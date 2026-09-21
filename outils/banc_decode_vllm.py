#!/usr/bin/env python3
"""Décodage vLLM à 12 séquences concurrentes, MÊME protocole que poste3
(outils/gpu/mesure/banc-horloge-decodage.py) : 12 invites, ctx 2048,
200 jetons décodés chacune, énergie NVML monotone (pas une moyenne de
puissances instantanées — voir energie.py), repos mesuré avant, net =
brut - repos×durée. Consigne de chef, duel A2, 14/09/2026.

    /opt/ia/vLLM/.venv/bin/python outils/banc_decode_vllm.py <dossier_modele>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gpu" / "mesure"))
from energie import Energie, repos  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

chemin_modele = sys.argv[1]
SLOTS = 12
CTX = 2048
N_JETONS = 200
VOCAB_APPROX = 150000


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
         enable_prefix_caching=False, gpu_memory_utilization=0.85,
         attention_config={"backend": "TRITON_ATTN"},
         max_model_len=CTX)
SP = SamplingParams(temperature=0.0, max_tokens=N_JETONS, ignore_eos=True)

prompts = [invite(1000 + k, min(256, CTX // 4)) for k in range(SLOTS)]

# Chauffe (compilation/allocation paresseuse hors mesure) — un lot court,
# comme poste3 sur acvram (prefill seul avant la fenetre).
llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)

base = repos(secondes=8.0)
with Energie() as e:
    sorties = llm.generate(prompts, SP, use_tqdm=False)

n = sum(len(s.outputs[0].token_ids) for s in sorties)
duree = e.duree
joules_net = max(e.joules - base.moyenne * duree, 0.0)
resultat = {
    "moteur": "vllm", "mode": "decodage", "modele": chemin_modele,
    "slots": SLOTS, "ctx": CTX, "n_jetons_decodes": n,
    "duree_mesure_s": round(duree, 4), "jetons_s": n / duree if duree else 0.0,
    "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
    "j_par_jeton_net": round(joules_net / n, 4) if n else None,
    "watts_repos": round(base.moyenne, 1),
    **e.resume(),
}
print("RESULTAT " + json.dumps(resultat, ensure_ascii=False))
