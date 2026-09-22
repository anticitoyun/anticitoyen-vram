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
# 5090 seule : le compteur d'énergie (energie.py) filtre les cartes sur cette variable ;
# sans elle il somme 5090 + 3080 Ti (constat de Sage sur la campagne 20 s, 14/09).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402

# Checkpoint ModelOpt NVFP4 du duel A2 (pas le converti acvram : vLLM y
# lirait des experts non quantifiés, 60 Gio, OOM à l'init).
chemin_modele = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3-Coder-30B-A3B-Instruct-FP4"
ENERGIE = int(os.environ.get("BANC_ENERGIE", "0"))   # pas mesurés au compteur NVML (0 = non)
SLOTS = int(os.environ.get("BANC_SLOTS", "12"))
CTX = 2048
VOCAB_APPROX = 150000
PAS = int(os.environ.get("BANC_PAS_NCU", "4"))


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


# BANC_ATTN : backend d'attention forcé (TRITON_ATTN = celui du duel Coder, attention
# classique) ; "auto" pour laisser vLLM choisir — obligatoire pour un modèle MLA
# (GLM-4.7-Flash : TRITON_MLA seul sur sm_120, forcer TRITON_ATTN lève « MLA not
# supported »). BANC_KV : kv_cache_dtype (auto | fp8).
ATTN = os.environ.get("BANC_ATTN", "TRITON_ATTN")
if os.environ.get("BANC_MLA_STAGES1") == "1":
    # sm_120 : 101 376 o de mémoire partagée par bloc ; le noyau Triton de décodage
    # MLA de vLLM 0.29 (triton_decode_attention.py:530) ne passe num_stages à 1 que
    # pour BLOCK_DMODEL >= 1024, et demande 102 400 o à BLOCK_DMODEL=512 (kv_lora 512
    # + rope 64 = Lk 576, GLM-4.7-Flash / DeepSeek) → OutOfResources à la capture.
    # Correctif d'expérience, borné à ce processus : même règle dès 512. Un correctif
    # d'installation (patch de /opt/ia/vLLM) est une décision utilisateur.
    import inspect
    import vllm.v1.attention.ops.triton_decode_attention as _tda
    src = inspect.getsource(_tda._decode_grouped_att_m_fwd)
    assert "BLOCK_DMODEL >= 1024" in src
    exec(src.replace("BLOCK_DMODEL >= 1024", "BLOCK_DMODEL >= 512"), _tda.__dict__)
    print("BANC_MLA_STAGES1 : num_stages=1 des BLOCK_DMODEL >= 512 (processus seul)")
opts = {} if ATTN == "auto" else {"attention_config": {"backend": ATTN}}
llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
          enable_prefix_caching=False, gpu_memory_utilization=0.85,
          kv_cache_dtype=os.environ.get("BANC_KV", "auto"), max_model_len=CTX, **opts)
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

if ENERGIE:
    # Même instrument que outils/energie_par_poste.py (Energie, 5090 seule) :
    # repos 30 s, puis ENERGIE pas de décodage b=SLOTS en boucle step().
    import time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "mesure"))
    from energie import Energie, nvml
    h = nvml().cartes[0][1]
    torch.cuda.synchronize(); time.sleep(2)
    with Energie(periode=0.5) as e0:
        time.sleep(float(os.environ.get("BANC_REPOS", "30")))
    sp2 = SamplingParams(temperature=0.0, max_tokens=ENERGIE + 8, ignore_eos=True)
    for k, p in enumerate(prompts):
        moteur.add_request(f"e{k}", TokensPrompt(prompt_token_ids=p), sp2)
    produits = {}
    while len(produits) < SLOTS:
        for out in moteur.step():
            if out.outputs and out.outputs[0].token_ids:
                produits[out.request_id] = 1
    torch.cuda.synchronize()
    horloges = []
    with Energie(periode=0.25) as e:
        t0 = time.perf_counter()
        for i in range(ENERGIE):
            moteur.step()
            if i % 10 == 0: horloges.append(nvml().horloge_sm(h))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    while moteur.has_unfinished_requests():
        moteur.step()
    import json
    print("ENERGIE " + json.dumps({"moteur": "vllm", "B": SLOTS, "pas": ENERGIE, "repos_W": round(e0.moyenne, 1),
        "W": round(e.moyenne, 1), "W_net": round(e.moyenne - e0.moyenne, 1), "ms_par_pas": round(1e3 * dt / ENERGIE, 2),
        "J_par_jeton_brut": round(e.joules / (ENERGIE * SLOTS), 4), "J_par_jeton_net": round((e.joules - e0.moyenne * e.duree) / (ENERGIE * SLOTS), 4),
        "MHz": sorted(horloges)[len(horloges) // 2], "bridages": sorted(e.bridages)}))
