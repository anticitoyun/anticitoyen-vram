#!/usr/bin/env python3
"""Pièce 74 : combien d'experts distincts par couche, à b = 12, chez vLLM et
chez nous — sur LES MÊMES ids d'invite, dans la même prise.

La pièce 73 conclut que vLLM tire 1,67 To/s de son MoE contre nos 1,37 « sur
les mêmes octets ». Ces octets supposent le même nombre d'experts distincts par
couche (33,4, mesuré chez NOUS seulement). Ce script le vérifie des deux côtés.

Crochet vLLM : `fused_topk` / `fused_topk_bias` enveloppées DANS ce processus —
l'installation de vLLM n'est pas touchée, et vLLM tourne hors ligne (un serveur
séparé ne verrait pas le crochet ; le routage ne dépend que des jetons).

Définition, identique des deux côtés : pour chaque appel de routage à M = 12
jetons, le nombre d'experts distincts parmi les M × top_k paires ; moyenne sur
les appels à M = 12 (les pas de décodage pleins). Les appels à M ≠ 12 sont
comptés à part et JAMAIS agrégés (alarme du verdict).

Usage : outils/carte.sh python outils/gpu/mesure/experts-distincts-p74.py vllm|acvram <alias> [--pas 40] [--json S]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

VOCAB = 151936


def invite(k: int, n: int) -> list[int]:
    """La même que `capture-godets.py` : déterministe, sans tokeniseur."""
    return [(k * 104729 + i * 7919) % (VOCAB - 100) + 10 for i in range(n)]


def resume(distincts: list[int], hors: list[int], top_k: int, b: int) -> dict:
    if not distincts:
        return {"appels_b": 0, "appels_hors_b": len(hors),
                "alarme": f"aucun appel à M = {b} : régime non représentatif"}
    return {"appels_b": len(distincts), "appels_hors_b": len(hors),
            "distincts_moy": round(statistics.mean(distincts), 2),
            "distincts_med": statistics.median(distincts),
            "distincts_min": min(distincts), "distincts_max": max(distincts),
            "paires": b * top_k}


def bras_vllm(alias: str, pas: int, b: int) -> dict:
    # Sans ceci, vLLM lance un sous-processus `EngineCore` et le crochet posé
    # ici ne le voit jamais (23/09 : quatre bras vides avant de le comprendre).
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.router import (fused_topk_bias_router,
                                                             fused_topk_router)
    distincts: list[int] = []
    hors: list[int] = []

    def enrober(mod, nom):
        vrai = getattr(mod, nom, None)
        if vrai is None:
            return
        def enrobee(*a, **kw):
            r = vrai(*a, **kw)
            ids = r[1] if isinstance(r, tuple) else r
            try:
                n = ids.shape[0]
                (distincts if n == b else hors).append(int(ids.unique().numel()))
            except Exception:                            # noqa: BLE001
                pass
            return r
        setattr(mod, nom, enrobee)

    enrober(fused_topk_router, "fused_topk")
    enrober(fused_topk_bias_router, "fused_topk_bias")
    # `enforce_eager` : la capture de graphes échouait
    # (`cudaErrorStreamCaptureInvalidated`) et elle ne change RIEN au routage —
    # les ids d'experts ne dépendent que des jetons. Déclaré au verdict.
    # TRITON_ATTN : en eager, vLLM choisit FlashInfer, dont le décodage xqa (sm120, KV fp8)
    # manque aux deux venvs (`RuntimeError: FlashInfer backend is not available`, 23/09) ;
    # l'attention ne touche le routage qu'à l'arrondi près.
    llm = LLM(model=alias, max_num_seqs=b, max_model_len=2304, gpu_memory_utilization=0.80,
              enforce_eager=True, disable_log_stats=True,
              attention_backend=os.environ.get("P74_ATTENTION", "TRITON_ATTN"))
    llm.generate([{"prompt_token_ids": invite(1000 + k, 256)} for k in range(b)],
                 SamplingParams(temperature=0.0, max_tokens=pas, ignore_eos=True))
    return resume(distincts, hors, 8, b)


def bras_acvram(alias: str, pas: int, b: int) -> dict:
    import torch
    sys.path.insert(0, os.getcwd())
    from acvram.engine import moe as M
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    distincts: list[int] = []
    hors: list[int] = []
    vrai = M.MoEBlock._forward_grouped

    def enrobee(self, x, topw, topi, eid=None):
        try:
            ids = topi if topi is not None else eid.view(-1, self.top_k)
            n = ids.shape[0]
            (distincts if n == b else hors).append(int(torch.unique(ids).numel()))
        except Exception:                                # noqa: BLE001
            pass
        return vrai(self, x, topw, topi, eid)

    M.MoEBlock._forward_grouped = enrobee
    charge = load_model(alias, dtype=torch.bfloat16, max_model_len=2304, max_concurrent_seqs=b)
    eng = Engine(charge, None, max_batch_size=b, max_model_len=2304, enable_cuda_graphs=True)
    for k in range(b):
        eng.add_request(invite(1000 + k, 256),
                        SamplingParams(temperature=0.0, max_tokens=pas + 8), request_id=f"d{k}")
    # préfill d'abord (M ≠ 12 : compté à part), puis `pas` pas de décodage pleins
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    for _ in range(pas):
        eng.step()
    return resume(distincts, hors, 8, b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("moteur", choices=["vllm", "acvram"])
    ap.add_argument("alias")
    ap.add_argument("--pas", type=int, default=40)
    ap.add_argument("-b", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    r = {"moteur": a.moteur, "alias": os.path.basename(a.alias), "b": a.b, "pas": a.pas}
    r.update((bras_vllm if a.moteur == "vllm" else bras_acvram)(a.alias, a.pas, a.b))
    print(json.dumps(r, ensure_ascii=False))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
