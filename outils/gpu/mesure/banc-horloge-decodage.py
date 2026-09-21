#!/usr/bin/env python3
"""Banc horloge SM x énergie — décodage Coder-30B, b=12.

Suite à poste7 (revue/poste7-veille-internet-13-09.md §1, arXiv 2605.11999 et
2501.08219) : au décodage, le bridage de puissance (400 W) est probablement
inerte, et verrouiller l'horloge SM plus bas recouvrerait de l'énergie à
débit quasi égal. Ce script mesure UN palier d'horloge à la fois — le
verrouillage lui-même (``nvidia-smi -lgc``, sudo) est fait par
``banc-horloge-decodage.sh``, qui appelle ce script après confirmation.

Ne verrouille RIEN ici : mesure seulement, avec ``outils/gpu/mesure/energie.py``
(compteur NVML monotone, pas une moyenne de puissances instantanées).

Usage :
    python banc-horloge-decodage.py MODEL_DIR PALIER SORTIE.json [--prefill]

    PALIER : étiquette du palier (« defaut », « 2400 », « 2100 », ... ) —
    n'AGIT sur rien, sert seulement à nommer le résultat et à relire
    l'horloge SM réellement observée pendant la fenêtre (contrôle du
    montage : la mesure doit confirmer le palier annoncé par l'appelant).
"""
import json
import os
import sys
import time

# Sans ceci, `import acvram` retombe sur le venv partage (copie hors
# worktree) plutot que sur ce checkout (piege du 13/09, cf.
# correctif-godet-hybride-13-09.md).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))                # outils/ -- regime.py

import torch

from energie import Energie, repos  # noqa: E402

MODEL_DIR = sys.argv[1]
PALIER = sys.argv[2]
SORTIE = sys.argv[3]
MODE_PREFILL = "--prefill" in sys.argv[4:]

SLOTS = 12
CTX = 2048
N_JETONS = 200          # decodage : assez long pour lisser le bruit d'energie
N_JETONS_PREFILL = 1024  # prefill : une seule passe longue, horloge haute seulement

os.environ["ACVRAM_HYBRID_SLOTS"] = str(SLOTS)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402
from regime import exiger_regime_nominal  # noqa: E402

print(f"[INFO] palier={PALIER} mode={'prefill' if MODE_PREFILL else 'decodage'} "
      f"modele={MODEL_DIR}", flush=True)
t0 = time.perf_counter()
loaded = load_model(MODEL_DIR, dtype=torch.bfloat16, max_model_len=CTX)
print(f"[INFO] charge en {time.perf_counter()-t0:.1f} s", flush=True)

vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000


def invite(k: int, n: int) -> list[int]:
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]


if MODE_PREFILL:
    # Prefill seul, horloge haute uniquement (chef) : une invite longue,
    # une seule sequence, mesure du temps + de l'energie de CETTE passe.
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=CTX,
                    enable_cuda_graphs=False)
    params = SamplingParams(temperature=0.0, max_tokens=1)
    prompt = invite(1, min(N_JETONS_PREFILL, CTX - 64))
    # chauffe (compilation/allocation paresseuse hors mesure)
    engine.add_request(invite(2, 64), SamplingParams(temperature=0.0, max_tokens=1))
    engine.step()
    torch.cuda.synchronize()
    exiger_regime_nominal(engine, autoriser_piles_inconnues=False)

    base = repos(secondes=5.0)
    with Energie() as e:
        engine.add_request(prompt, params)
        t_p0 = time.perf_counter()
        engine.step()          # une seule passe de prefill (invite non decoupee)
        torch.cuda.synchronize()
        t_p1 = time.perf_counter()
    duree = t_p1 - t_p0
    n = len(prompt)
    joules_net = max(e.joules - base.moyenne * e.duree, 0.0)
    resultat = {
        "mode": "prefill", "palier": PALIER, "modele": MODEL_DIR,
        "n_jetons_invite": n, "duree_mesure_s": round(duree, 4), "jetons_s": n / duree if duree else 0.0,
        "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
        "j_par_jeton_net": round(joules_net / n, 4) if n else None,
        "watts_repos": round(base.moyenne, 1),
        **e.resume(),
    }
else:
    engine = Engine(loaded, None, max_batch_size=SLOTS, max_model_len=CTX)
    engine.warm_graphs()
    exiger_regime_nominal(engine, autoriser_piles_inconnues=False)
    torch.cuda.reset_peak_memory_stats(0)
    params = SamplingParams(temperature=0.0, max_tokens=N_JETONS)
    for k in range(SLOTS):
        engine.add_request(invite(1000 + k, min(256, CTX // 4)), params, request_id=f"d{k}")

    base = repos(secondes=8.0)
    n_avant = engine.stats.decode_tokens
    with Energie() as e:
        t_d0 = time.perf_counter()
        n_pas = 0
        while engine.running or engine.waiting:
            engine.step()
            n_pas += 1
            if n_pas > N_JETONS + 20:
                raise RuntimeError("le lot ne se termine pas")
        torch.cuda.synchronize()
        t_d1 = time.perf_counter()
    duree = t_d1 - t_d0
    n = engine.stats.decode_tokens - n_avant
    joules_net = max(e.joules - base.moyenne * e.duree, 0.0)
    resultat = {
        "mode": "decodage", "palier": PALIER, "modele": MODEL_DIR,
        "slots": SLOTS, "ctx": CTX, "n_jetons_decodes": n,
        "duree_mesure_s": round(duree, 4), "jetons_s": n / duree if duree else 0.0,
        "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
        "j_par_jeton_net": round(joules_net / n, 4) if n else None,
        "watts_repos": round(base.moyenne, 1),
        **e.resume(),
    }

print(json.dumps(resultat, indent=2, ensure_ascii=False), flush=True)
with open(SORTIE, "w", encoding="utf-8") as fh:
    json.dump(resultat, fh, indent=2, ensure_ascii=False)
print(f"[INFO] resultats -> {SORTIE}", flush=True)

if resultat.get("invalidations", "aucune") != "aucune":
    print(f"[ATTENTION] fenetre invalidee : {resultat['invalidations']}", flush=True)
