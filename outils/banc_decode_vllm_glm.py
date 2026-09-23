#!/usr/bin/env python3
"""Bras vLLM du duel MLA apparié (GLM-4.7-Flash, glm4_moe_lite) : décodage à
b = 1 / 4 / 12 séquences, mêmes dénominateurs que outils/banc_decode_vllm.py
(jetons décodés / durée de la fenêtre, énergie NVML monotone, repos mesuré
avant, net = brut − repos×durée), 5090 SEULE (CUDA_VISIBLE_DEVICES=0 posé
ici : sans lui energie.py somme les deux cartes — Sage, 14/09), fenêtre
≥ 20 s au compteur (lots successifs de 1 024 jetons décodés par séquence
jusqu'à 20 s ; le prefill de 256 jetons par lot pèse < 2 % de la fenêtre,
compté et publié).

Checkpoint : GadflyII/GLM-4.7-Flash-NVFP4 (compressed-tensors NVFP4 : experts
et MLP dense en E2M1 bloc 16, attention MLA / lm_head / routeur en bf16),
20,4 Go — le bf16 en fp8 dynamique ferait 31 Go de poids, hors 5090. MLA sur
sm_120 : seul backend TRITON_MLA (platforms/cuda.py:129), KV fp8 accepté
(triton_mla.py:132), graphes CUDA en décodage (UNIFORM_SINGLE_TOKEN_DECODE).

    outils/carte.sh /opt/ia/vLLM/.venv/bin/python outils/banc_decode_vllm_glm.py [dossier_modele]
    BANC_SLOTS=1,4,12  BANC_KV=fp8|auto  BANC_FENETRE_S=20  BANC_JETONS=1024
"""
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ICI, "gpu", "mesure"))
from energie import Energie, repos  # noqa: E402
from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402

if os.environ.get("BANC_MLA_STAGES1", "1") == "1":
    # sm_120 : 99 Ko de shared par bloc ; le décodage MLA Triton de vLLM 0.29 demande
    # 102 400 o à BLOCK_DMODEL=512 (Lk 576) faute de la garde num_stages=1 réservée à
    # >= 1024 (triton_decode_attention.py:530, écrite pour H100). Même règle dès 512,
    # dans ce processus seulement — voir revue/duel-mla-glm-14-09.md.
    import inspect
    import vllm.v1.attention.ops.triton_decode_attention as _tda
    _src = inspect.getsource(_tda._decode_grouped_att_m_fwd)
    assert "BLOCK_DMODEL >= 1024" in _src
    exec(_src.replace("BLOCK_DMODEL >= 1024", "BLOCK_DMODEL >= 512"), _tda.__dict__)

chemin_modele = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/4TO_SATACMR_2022/Modeles/models_vllm/GLM-4.7-Flash-NVFP4"
SLOTS_LISTE = [int(x) for x in os.environ.get("BANC_SLOTS", "1,4,12").split(",")]
CTX = int(os.environ.get("BANC_CTX", "2048"))
N_JETONS = int(os.environ.get("BANC_JETONS", "1024"))
FENETRE = float(os.environ.get("BANC_FENETRE_S", "20"))
KV = os.environ.get("BANC_KV", "fp8")
VOCAB_APPROX = 150000
INVITE = min(256, CTX // 4)


def invite(k: int, n: int) -> list:
    return [(k * 104729 + i * 7919) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


llm = LLM(model=chemin_modele, dtype="auto", enforce_eager=False,
          enable_prefix_caching=False, gpu_memory_utilization=0.85,
          kv_cache_dtype=KV, max_model_len=CTX, max_num_seqs=max(SLOTS_LISTE))
SP = SamplingParams(temperature=0.0, max_tokens=N_JETONS, ignore_eos=True)

# Chauffe : un lot complet au plus grand b (graphes capturés pour tous les godets).
chauffe = [TokensPrompt(prompt_token_ids=invite(5000 + k, INVITE)) for k in range(max(SLOTS_LISTE))]
llm.generate(chauffe, SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)

for slots in SLOTS_LISTE:
    prompts = [TokensPrompt(prompt_token_ids=invite(1000 + k, INVITE)) for k in range(slots)]
    base = repos(secondes=8.0)
    n = 0; lots = 0
    with Energie() as e:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < FENETRE:
            sorties = llm.generate(prompts, SP, use_tqdm=False)
            n += sum(len(s.outputs[0].token_ids) for s in sorties); lots += 1
    duree = e.duree
    joules_net = max(e.joules - base.moyenne * duree, 0.0)
    print("RESULTAT " + json.dumps({
        "moteur": "vllm", "mode": "decodage", "modele": chemin_modele, "kv_cache_dtype": KV,
        "slots": slots, "ctx": CTX, "invite": INVITE, "jetons_par_seq": N_JETONS, "lots": lots,
        "prefill_jetons": lots * slots * INVITE, "n_jetons_decodes": n,
        "duree_mesure_s": round(duree, 3), "fenetre_valide": duree >= FENETRE,
        "jetons_s": round(n / duree, 1) if duree else 0.0,
        "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
        "j_par_jeton_brut": round(e.joules / n, 4) if n else None,
        "j_par_jeton_net": round(joules_net / n, 4) if n else None,
        "watts_repos": round(base.moyenne, 1), **e.resume()}, ensure_ascii=False), flush=True)
