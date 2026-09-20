#!/usr/bin/env python3
"""Fréquence d'activation des experts GLM — résidence dynamique, ordre poste7
(revue/poste7-avis-exterieur-16-09.md § point 2 ; repris le 17/09,
revue/poste7-plan-completion-comparatif-17-09.md § 3), à sec.

Mesure : sur un corpus réel, un seul prefill dense (une passe, tous les
jetons), lit `MoEBlock._usage_routage` (histogramme par expert, toujours
actif, `model.py:615`) après coup — aucune sonde à poser, le compteur
existe déjà (chantier colibrì). Par couche MoE : fraction de l'activité
de routage portée par les 50 % d'experts les plus sollicités.

Scellé (poste7, 16/09) : attendu ≥ 80 % ; réfuté si < 60 % sur 50 % des
experts (la résidence dynamique ne vaudrait rien sur ce MoE).

À sec, aucun GPU.
"""
import os
import sys
import time
from pathlib import Path
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

MODEL = _RACINE + "/GLM-4.7-Flash-srcbf16-nvfp4"
CORPUS = "/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt"
N_JETONS = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MAX_MODEL_LEN = N_JETONS + 64


def main() -> int:
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer
    from acvram.evaluate import _load_corpus

    tokenizer = load_tokenizer(MODEL)
    texte = _load_corpus(CORPUS)
    ids = tokenizer.encode(texte)[:N_JETONS]
    print(f"corpus : {len(ids)} jetons", flush=True)

    t0 = time.perf_counter()
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=MAX_MODEL_LEN,
                        device_override="cpu")
    print(f"chargé en {time.perf_counter()-t0:.1f} s", flush=True)

    engine = Engine(loaded, tokenizer, max_batch_size=1, max_model_len=MAX_MODEL_LEN,
                    enable_cuda_graphs=False)
    t0 = time.perf_counter()
    engine.add_request(ids, SamplingParams(temperature=0.0, max_tokens=1),
                       request_id="s0")
    while engine.running or engine.waiting:
        engine.step()
    print(f"prefill exécuté en {time.perf_counter()-t0:.1f} s", flush=True)

    from acvram.engine.model import MoEBlock
    resultats = []
    for i, layer in enumerate(loaded.model.layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, MoEBlock):
            continue
        hist = mlp._usage_routage
        if hist is None:
            print(f"  couche {i} : MoE mais aucun histogramme (pas atteinte ?)",
                 flush=True)
            continue
        h = hist.to(torch.float64)
        total = float(h.sum())
        if total == 0:
            continue
        e = h.numel()
        moitie = (e + 1) // 2
        triees, _ = torch.sort(h, descending=True)
        part_moitie = float(triees[:moitie].sum()) / total
        resultats.append({"couche": i, "n_experts": e, "moitie": moitie,
                          "total_routages": total, "part_top_moitie": part_moitie})
        print(f"  couche {i:2d} : top {moitie}/{e} experts portent "
             f"{part_moitie:.2%} du routage (total={total:.0f})", flush=True)

    if not resultats:
        print("AUCUNE couche MoE mesurée — corpus trop court ou modèle inattendu")
        return 1

    parts = [r["part_top_moitie"] for r in resultats]
    moyenne = sum(parts) / len(parts)
    mediane = sorted(parts)[len(parts) // 2]
    pire = min(parts)
    print(f"\n{len(resultats)} couches MoE mesurées")
    print(f"part top-50% : moyenne={moyenne:.2%} médiane={mediane:.2%} pire={pire:.2%}")

    if moyenne >= 0.80:
        verdict = "CONFORME à l'attendu (≥80%)"
    elif moyenne >= 0.60:
        verdict = "au-dessus du seuil de réfutation (≥60%) mais sous l'attendu (80%) — chantier retenu, gain à revoir à la baisse"
    else:
        verdict = "RÉFUTÉ (<60%) — la résidence dynamique ne vaut rien sur ce MoE"
    print(f"VERDICT (moyenne des couches) : {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
