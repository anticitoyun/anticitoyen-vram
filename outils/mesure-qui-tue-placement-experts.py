#!/usr/bin/env python3
"""Mesure qui tue (bead anticitoyen-vram-pds, point 4 — poste7 §4).

VRAM artificiellement bornée par « plan forcé » (chef) : pas de carte plus
petite, pas de nouvelle variable de plafond mémoire — on réutilise le levier
qui existe déjà pour l'exil par couche (`ACVRAM_EXIL_COUCHES`,
`engine/loader.py:_forcer_exil`) et son nouveau pendant par expert
(`ACVRAM_EXIL_EXPERTS_FRACTION`, `_forcer_exil_experts`).

MÊME volume d'octets exilé dans les deux régimes, condition nécessaire pour
que la comparaison porte sur le GRAIN de l'exil et non sur sa quantité :
Qwen3-Coder-30B-A3B-nvfp4 a 48 couches MoE à 128 experts chacune, UNIFORME
(vérifié sur `acvram_manifest.json` avant d'écrire ce script) — donc exiler
24 couches entières libère exactement les mêmes octets qu'exiler 50 % des
experts sur les 48 couches.

Trois régimes, deux lots (b=1, b=12) : résident complet (référence), exil PAR
COUCHE (aujourd'hui), exil PAR EXPERT (ce chantier). Facteur = débit résident
/ débit exilé. Cible du chantier : facteur < 1,5 pour l'exil par expert, là où
l'exil par couche vaut 3-4 (mémoire : « l'exil d'une couche est une falaise »).

Usage :
    outils/carte.sh .venv/bin/python outils/mesure-qui-tue-placement-experts.py \\
        --model "$(outils/racine_modeles.py)"/Qwen3-Coder-30B-A3B-nvfp4 \\
        --sortie mesure-qui-tue-13-09.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

N_COUCHES_MOE = 48
FRACTION_EXILEE = 0.5
N_COUCHES_EXIL_EQUIVALENT = round(N_COUCHES_MOE * FRACTION_EXILEE)  # 24

CTX = 2048
N_JETONS = 200
PROMPT_LEN = 256

REGIMES = {
    "resident_complet": None,
    "exil_par_couche": ("ACVRAM_EXIL_COUCHES", str(N_COUCHES_EXIL_EQUIVALENT)),
    "exil_par_expert": ("ACVRAM_EXIL_EXPERTS_FRACTION", str(FRACTION_EXILEE)),
}


def _invite(vocab: int, k: int, n: int) -> list[int]:
    return [(k * 104729 + i * 7919) % (vocab - 100) + 10 for i in range(n)]


def _charger(model_dir: str, env_exil):
    import torch

    for var, _ in filter(None, [REGIMES["exil_par_couche"], REGIMES["exil_par_expert"]]):
        os.environ.pop(var, None)
    if env_exil:
        os.environ[env_exil[0]] = env_exil[1]
    # REPIN coupé : ce banc mesure le placement STATIQUE, pas le va-et-vient
    # dynamique. Le laisser actif casse ici pour une raison propre au banc
    # (deux Engine successifs sur le MEME `loaded` — `Engine._pin` se
    # réinitialise depuis `m._pin_experts`, un instantané pris au chargement,
    # que le premier Engine ne remet jamais à jour en échangeant réellement
    # les poids : le second Engine part alors d'un pin périmé et peut choisir
    # de « promouvoir » un expert déjà résident -> AttributeError sur
    # `lin.streamed.host`. Latent, mais sans risque en service réel : un
    # SEUL Engine y vit pour toute la durée du modèle chargé.
    os.environ["ACVRAM_REPIN"] = "0"

    from acvram.engine.loader import load_model

    t0 = time.perf_counter()
    loaded = load_model(model_dir, dtype=torch.bfloat16, max_model_len=CTX)
    load_s = time.perf_counter() - t0
    return loaded, load_s


def _mesurer(loaded, batch: int) -> dict:
    import torch

    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
    engine = Engine(loaded, None, max_batch_size=batch, max_model_len=CTX)
    engine.warm_graphs()

    params = SamplingParams(temperature=0.0, max_tokens=N_JETONS)
    for k in range(batch):
        engine.add_request(_invite(vocab, 2000 + k, PROMPT_LEN), params,
                           request_id=f"d{k}")

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

    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return {"batch": batch, "n_jetons_decodes": n, "duree_s": round(duree, 4),
            "jetons_s": round(n / duree, 2) if duree else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sortie", default="mesure-qui-tue.json")
    ap.add_argument("--regime", choices=sorted(REGIMES), default=None,
                    help="un seul régime (un processus par régime, moins de "
                         "pression mémoire) ; omis = tous, dans le même "
                         "processus")
    args = ap.parse_args()

    import torch

    # Un régime par processus : chaque charge/mesure/décharge complètement
    # avant que le processus suivant démarre — plus robuste qu'un del +
    # empty_cache() interne quand la machine est déjà sous pression (mesuré
    # le 13/09 : deux tués coup sur coup en tout-en-un, aucun kernel OOM dans
    # dmesg -- la garde qui a tué venait de l'orchestrateur, pas du noyau).
    regimes = {args.regime: REGIMES[args.regime]} if args.regime else REGIMES

    resultats: dict = {}
    if os.path.isfile(args.sortie):
        with open(args.sortie, "r", encoding="utf-8") as fh:
            resultats = json.load(fh).get("resultats", {})

    for nom, env in regimes.items():
        print(f"[INFO] chargement regime={nom} ({env})", flush=True)
        loaded, load_s = _charger(args.model, env)
        resultats[nom] = {"charge_s": round(load_s, 1)}
        for b in (1, 12):
            print(f"[INFO]  regime={nom} b={b}", flush=True)
            r = _mesurer(loaded, b)
            resultats[nom][f"b{b}"] = r
            print(f"[INFO]   -> {r['jetons_s']} jetons/s", flush=True)
        del loaded
        gc.collect()
        torch.cuda.empty_cache()

    facteurs: dict = {}
    if "resident_complet" in resultats:
        for b in (1, 12):
            ref = resultats["resident_complet"][f"b{b}"]["jetons_s"]
            for nom in ("exil_par_couche", "exil_par_expert"):
                if nom in resultats:
                    d = resultats[nom][f"b{b}"]["jetons_s"]
                    facteurs.setdefault(nom, {})[f"b{b}"] = (
                        round(ref / d, 3) if d else None)

    sortie = {
        "modele": args.model,
        "n_couches_moe": N_COUCHES_MOE,
        "fraction_exilee": FRACTION_EXILEE,
        "n_couches_exil_equivalent": N_COUCHES_EXIL_EQUIVALENT,
        "resultats": resultats,
        "facteurs_vs_resident": facteurs,
    }
    print(json.dumps(sortie, indent=2, ensure_ascii=False), flush=True)
    with open(args.sortie, "w", encoding="utf-8") as fh:
        json.dump(sortie, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] resultats -> {args.sortie}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
