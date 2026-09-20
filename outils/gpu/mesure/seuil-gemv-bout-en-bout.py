#!/usr/bin/env python3
"""Le seuil `_NVFP4_GEMV_MAX` : 8 ou 32, mesure de bout en bout.

Deux mesures se contredisent sur le meme regime, et ce banc les departage.

  - `kernels/__init__.py:395` : « Balaye sur un dense de 27B, le TTFT d une
    invite de 16 jetons vaut 194,7 ms a 8, 202,8 a 32 — monter le seuil ne fait
    que perdre. »
  - Banc de noyaux du 10/09 : forcer le GEMV au-dela de 8 rend **3,2 a 6,4x**
    plus vite a douze lignes sur six formes du parc, et **+27,8 dB** de
    justesse contre le poids reellement stocke.

Une constante posee par une mesure ne se change pas sur la foi d une autre
mesure d un autre perimetre. Celui-ci a le meme perimetre que celle qui a pose
le 8 : un modele entier, un TTFT — plus le decodage concurrent, que l ancienne
n avait pas.

**Une valeur par processus.** Le seuil est lu a l import ; et deux valeurs dans
un meme processus partageraient un cache d allocateur deja faconne par la
premiere. L ABBA se fait en enchainant quatre lancements A B B A, pas en
alternant dans une boucle.
"""
import argparse
import os
import sys
import time
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

A = _RACINE


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modele", default="Qwen3-4B-srcgguf-nvfp4")
    ap.add_argument("--seqs", type=int, default=12)
    ap.add_argument("--invite", type=int, default=16)
    ap.add_argument("--jetons", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--sans-graphes", action="store_true",
                    help="le seul regime ou le backend 110 est pris")
    ns = ap.parse_args(argv[1:])

    if torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES=0 obligatoire")
    # Le seuil REELLEMENT en vigueur, pas le defaut suppose : lire
    # `os.environ.get(..., "8")` publiait « 8 » alors que le noyau tournait a
    # 32. Un instrument qui etiquette un bras par une valeur qu il n a pas
    # mesuree est pire que pas d etiquette.

    from acvram.kernels import _NVFP4_GEMV_MAX
    seuil = _NVFP4_GEMV_MAX
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.layers import QuantLinear

    charge = load_model(os.path.join(A, ns.modele), max_model_len=ns.max_model_len)
    moteur = Engine(charge, None, max_batch_size=ns.seqs,
                    max_model_len=ns.max_model_len,
                    enable_cuda_graphs=not ns.sans_graphes)
    exiles = sum(1 for m in charge.model.modules()
                 if isinstance(m, QuantLinear) and m.streamed is not None)
    g_ = getattr(moteur, "graphs", None)
    graphes = bool(getattr(g_, "enabled", False))

    params = SamplingParams(temperature=0.0, max_tokens=ns.jetons)
    for i in range(ns.seqs):
        ids = [(i * 7919 + j * 31 + 11) % 30000 + 1 for j in range(ns.invite)]
        moteur.add_request(ids, params, request_id=f"r{i}")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    moteur.step()                       # le pas qui contient le prefill
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000.0

    # Decodage : on compte les JETONS REELLEMENT PRODUITS, pas les pas demandes.
    t1 = time.perf_counter()
    pas = jetons = 0
    ids_produits = []
    while moteur.running or moteur.waiting:
        sorties = moteur.step()
        pas += 1
        for s_ in (sorties or []):
            ids_produits.extend(list(getattr(s_, "token_ids", ()) or ()))
        jetons += sum(len(getattr(s, "token_ids", ()) or ()) for s in (sorties or []))
        if pas > ns.jetons + ns.invite + 16:
            break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    debit_pas = pas / dt if dt else 0.0

    # « graphes actifs » ne dit PAS qu ils ont servi : `graphs.run` rend None
    # et retombe en eager sur longueurs mixtes, sur cle inconnue au-dela de
    # MAX_GRAPHS, ou apres un echec de capture. Le seul chiffre qui tranche est
    # le nombre de REJEUX rapporte au nombre de pas.
    rej = getattr(g_, "replays", 0) if g_ is not None else 0
    cap = getattr(g_, "captures", 0) if g_ is not None else 0
    # Le TEXTE, pas seulement le debit : un chemin plus rapide qui repond
    # autre chose n a pas gagne. Empreinte des jetons reellement produits.
    import hashlib
    sig = (hashlib.sha256(repr(ids_produits).encode()).hexdigest()[:12]
           if ids_produits else "-")
    print(f"{seuil}\t{ttft:.1f}\t{debit_pas:.2f}\t{pas}\t{jetons}\t"
          f"{exiles}\t{'oui' if graphes else 'NON'}\t{rej}/{pas}\t{cap}\t{sig}\t"
          f"{os.environ.get('ACVRAM_DISABLE_FP4_TC','') and 'sans110' or 'avec110'}")
    return 0


if __name__ == "__main__":
    print("seuil\tttft_ms\tpas_s\tpas\tjetons\texiles\tgraphes", file=sys.stderr)
    sys.exit(main(sys.argv))
