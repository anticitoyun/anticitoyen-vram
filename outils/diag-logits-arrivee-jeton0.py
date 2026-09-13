#!/usr/bin/env python3
"""Bead runner (14/09 soir) : poste4 a trouve que sa sequence "arrivee"
(admise en cours de lot, indice 11) diverge d'eager des son PREMIER jeton
decode -- contrairement a s4/s7 (bruit bf16 deja etabli), une divergence au
tout premier pas d'un slot repris est plus suspecte d'un vrai bogue
d'admission (table/position/slot_mapping). Controle de chef : ecart
top1/top2 au point de divergence, sur les DEUX chemins.

UN SEUL chargement par PROCESSUS (le double chargement dans le meme
processus a deja fait tourner un plan degrade -- exil, graphes coupes,
modele reparti sur 2 cartes, cf. mesure-pipeline-ab.py/14-09) :

    outils/carte.sh outils/diag-logits-arrivee-jeton0.sh
"""
import json
import os
import sys

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)
import importlib.util
os.environ["ACVRAM_REPIN"] = "0"

import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams

_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
MODEL = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))
N_SEQ_INIT = 11


def invite(k, rep=0, n=64):
    # Meme formule EXACTE que outils/test_graphes_vs_eager.py (`ajouter`) :
    # un autre generateur de prompt trouverait une autre bascule bf16 (ou
    # aucune), pas forcement celle que poste4 a signalee.
    return [(1000 + rep * 7919 + k * 101 + j * 13) % 150000 + 10 for j in range(n)]


def capturer(graphes: bool) -> dict:
    engine = Engine(load_model(MODEL, dtype=torch.bfloat16, device_override="cuda:0"),
                    None, max_batch_size=12, max_model_len=1024,
                    enable_cuda_graphs=graphes, enable_prefix_cache=False)
    engine._eos = set()
    maxs = [60, 200, 90, 200, 120, 160, 200, 45, 200, 75, 200, 200]
    for i in range(N_SEQ_INIT):
        engine.add_request(invite(i), SamplingParams(temperature=0.0, max_tokens=maxs[i]),
                           request_id=f"s{i}")

    capture: dict = {}
    orig = engine._emit

    def emit_espion(logits, seqs):
        # `request_id`, pas `seq.id` capture avant coup : plus direct, pas
        # d'etat intermediaire a desynchroniser. `step()` prefille ET
        # decode s11 dans le MEME appel (`_decodables()` l'inclut des que
        # `prefilled` passe a vrai, ligne 787 juste apres son prefill) :
        # DEUX appels a `_emit` pour s11 dans un seul `engine.step()`,
        # jeton 0 (prefill) puis jeton 1 (decode) -- ne garder QUE le
        # premier, sous peine de capturer jeton 1 en pensant lire jeton 0
        # (bogue trouve ce soir sur la premiere version du script).
        if "vals" not in capture:
            for i, seq in enumerate(seqs):
                if seq.request_id == "s11":
                    top2 = torch.topk(logits[i:i + 1].to(torch.float32), 2, dim=-1)
                    capture["vals"] = top2.values[0].tolist()
                    capture["idx"] = top2.indices[0].tolist()
                    capture["pas"] = pas_actuel["v"]
                    break
        return orig(logits, seqs)
    engine._emit = emit_espion

    pas_actuel = {"v": -1}
    admise = False
    pas = 0
    plafond = 60  # 30 pour amener au pas d'admission + marge large, jamais 200+
    while (engine.running or engine.waiting) and pas < plafond:
        if pas == 30 and not admise:
            engine.add_request(invite(11), SamplingParams(temperature=0.0, max_tokens=200),
                               request_id="s11")
            admise = True
        pas_actuel["v"] = pas
        engine.step()
        pas += 1
        if "vals" in capture:
            break
    if "vals" not in capture:
        raise RuntimeError(f"jeton 0 de s11 jamais capture en {pas} pas (plafond {plafond})")
    return capture


if __name__ == "__main__":
    graphes = sys.argv[1] == "graphes"
    cap = capturer(graphes)
    # Le chargeur imprime sur stdout (messages de plan/exil) : la sortie
    # exploitable va dans un fichier, jamais melangee au reste de stdout.
    with open(sys.argv[2], "w") as f:
        json.dump(cap, f)
    print(f"capture ecrite dans {sys.argv[2]} : {cap}", flush=True)
