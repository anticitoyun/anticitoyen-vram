#!/usr/bin/env python3
"""Pas de décodage vLLM à 12 séquences dans une plage NVTX « mesure », pour
ncu (--nvtx --nvtx-include "mesure/"). Même moteur et mêmes invites que
outils/banc_decode_vllm.py (duel A2), mais EngineCore DANS le processus
(VLLM_ENABLE_V1_MULTIPROCESSING=0, posé ici) et boucle step() à la main :
prefill hors plage, puis BANC_PAS_NCU pas de décodage dans la plage.

    ncu --nvtx --nvtx-include "mesure/" ... \
      /opt/ia/vLLM/.venv/bin/python outils/ncu_vllm_decode12.py <dossier_modele>
"""
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402

chemin_modele = sys.argv[1]
SLOTS = int(os.environ.get("BANC_SLOTS", "12"))
CTX = 2048
VOCAB_APPROX = 150000
PAS = int(os.environ.get("BANC_PAS_NCU", "4"))


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
          enable_prefix_caching=False, gpu_memory_utilization=0.85,
          attention_config={"backend": "TRITON_ATTN"}, max_model_len=CTX)
moteur = llm.llm_engine
prompts = [invite(1000 + k, min(256, CTX // 4)) for k in range(SLOTS)]

# Chauffe : un lot complet, graphes capturés et rejoués une fois.
llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
             SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True), use_tqdm=False)

sp = SamplingParams(temperature=0.0, max_tokens=PAS + 8, ignore_eos=True)
for k, p in enumerate(prompts):
    moteur.add_request(f"m{k}", TokensPrompt(prompt_token_ids=p), sp)
# Prefill (éventuellement en tranches) : on avance jusqu'à ce que chaque
# séquence ait produit son premier jeton — les pas suivants sont du pur
# décodage à b=SLOTS.
produits = {}
while len(produits) < SLOTS:
    for out in moteur.step():
        if out.outputs and out.outputs[0].token_ids:
            produits[out.request_id] = len(out.outputs[0].token_ids)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("mesure")
for _ in range(PAS):
    moteur.step()
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
while moteur.has_unfinished_requests():
    moteur.step()
print("NCU_OK", SLOTS, PAS)
