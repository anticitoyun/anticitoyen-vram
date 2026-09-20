#!/usr/bin/env python3
"""A/B `ACVRAM_PIPELINE=0` vs `1`, résident complet Coder-30B b=12 (bead
runner, 14/09). Témoin : 681 j/s (masquage des créneaux fantômes,
acvram-memoire/revue/prediction-masquage-fantomes-14-09.md). Prédiction
scellée du chantier (chef) : pas ≤ 13,5 ms (16,5 → ≤13,5).

Usage :
    outils/carte.sh .venv/bin/python outils/mesure-pipeline-ab.py
"""
import os
import time

import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from regime import exiger_regime_nominal          # sibling de outils/, pas un paquet
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
N_SLOTS = 12
N_JETONS = 200
PROMPT_LEN = 256


def _invite(vocab: int, k: int, n: int) -> list[int]:
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]


def _mesurer(pipeline: bool) -> dict:
    os.environ["ACVRAM_PIPELINE"] = "1" if pipeline else "0"
    os.environ["ACVRAM_REPIN"] = "0"

    t0 = time.perf_counter()
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=2048)
    load_s = time.perf_counter() - t0

    vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
    engine = Engine(loaded, None, max_batch_size=N_SLOTS, max_model_len=2048)
    engine.pipeline_actif = pipeline
    engine.warm_graphs()
    exiger_regime_nominal(engine, autoriser_piles_inconnues=False)

    params = SamplingParams(temperature=0.0, max_tokens=N_JETONS)
    for k in range(N_SLOTS):
        engine.add_request(_invite(vocab, 2000 + k, PROMPT_LEN), params, request_id=f"d{k}")

    n_avant = engine.stats.decode_tokens
    t_d0 = time.perf_counter()
    n_pas = 0
    while engine.running or engine.waiting:
        engine.step()
        n_pas += 1
        if n_pas > N_JETONS + 20:
            raise RuntimeError("le lot ne se termine pas")
    torch.cuda.synchronize()
    duree = time.perf_counter() - t_d0
    n = engine.stats.decode_tokens - n_avant

    del engine, loaded
    torch.cuda.empty_cache()

    return {"pipeline": pipeline, "charge_s": round(load_s, 1),
            "n_jetons": n, "duree_s": round(duree, 4),
            "jetons_s": round(n / duree, 2) if duree else 0.0,
            "ms_par_pas": round(duree * 1000 / n_pas, 3) if n_pas else None}


def main() -> None:
    import sys
    # `--seul 0|1` : un seul bras, DANS UN PROCESSUS À LUI -- charger les
    # deux modeles dans le meme processus (comportement historique) laisse
    # une deuxieme charge dans un etat VRAM non totalement recupere par
    # `empty_cache()` (constate le 14/09 soir : plan degrade, graphes CUDA
    # desactives au deuxieme chargement) -- confondu qui biaise TOUJOURS
    # contre PIPELINE=1 (mesure en second). outils/mesure-pipeline-ab.sh
    # lance les deux bras en deux processus separes et compare.
    if "--seul" in sys.argv:
        i = sys.argv.index("--seul")
        r = _mesurer(pipeline=bool(int(sys.argv[i + 1])))
        print(r, flush=True)
        return

    r0 = _mesurer(pipeline=False)
    print(f"[A] PIPELINE=0 : {r0}", flush=True)
    r1 = _mesurer(pipeline=True)
    print(f"[B] PIPELINE=1 : {r1}", flush=True)

    gain = (r1["jetons_s"] / r0["jetons_s"] - 1) * 100 if r0["jetons_s"] else None
    print(f"\ngain débit A->B : {gain:+.1f} %" if gain is not None else "gain : indéterminé",
          flush=True)
    print(f"témoin (masquage seul) : 681 j/s", flush=True)
    print(f"prédiction scellée : pas <= 13,5 ms (B={r1['ms_par_pas']} ms)", flush=True)
    if r1["ms_par_pas"] is not None:
        verdict = "TIENT" if r1["ms_par_pas"] <= 13.5 else "NE TIENT PAS"
        print(f"verdict prédiction : {verdict}", flush=True)


if __name__ == "__main__":
    main()
