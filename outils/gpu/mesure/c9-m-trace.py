"""C9 M-TRACE (poste1-c9-conception-21-09 § 3.2) : taux de succès d un cache
d experts ÉPINGLÉ à la capacité du 119B (C = 52 experts par couche = 41 % de
128), mesuré sur une trace de routage réelle — proxy Qwen3-Coder-30B nvfp4
(top-8 de 128, 48 couches) tant que le 119B ne charge pas ; sur le 119B dès
qu il charge (même commande, `--experts 128`, C inchangé).

Politique jugée : `pin` (les C experts les plus demandés par couche, appris
sur la première moitié de la trace, jugés sur la seconde — jamais sur la
moitié qui l a appris, `trace_routage.taux_de_succes_pin`) ; `lru` en
témoin. Prédit (écrit avant) : h_pin(52) global 0,55 ± 0,10. Réfuté sous
0,45 : le cache n engage pas, S3 ≈ S2 dans la note ; au-dessus de 0,65 : les
bornes S1/S3 se recalculent à la valeur (mieux).

Usage : outils/carte.sh python outils/gpu/mesure/c9-m-trace.py --model DIR [--jetons 5000]
        [--capacite 52] [--experts 128] [--sortie trace.txt] [--json r.json]
     ou (sans carte) : c9-m-trace.py --trace trace.txt [--capacite 52] [--experts 128]
Carte : ≤ 10 min (chargement + 5 000 jetons b=1 sans graphes, la trace synchronise).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))

PREDIT, TOLERANCE, SEUIL_REFUTE = 0.55, 0.10, 0.45


def analyser(trace: str, capacite: int, nb_experts: int, entrainement: float = 0.5) -> dict:
    from acvram.memory.trace_routage import (relire, taux_de_succes_lru_juge,
                                             taux_de_succes_pin)
    n_lignes, jetons, couches = 0, set(), set()
    for jeton, couche, _ in relire(trace):
        n_lignes += 1
        jetons.add(jeton)
        couches.add(couche)
    pin = taux_de_succes_pin(trace, [capacite], entrainement)[capacite]
    lru = taux_de_succes_lru_juge(trace, capacite, entrainement)
    h = float(pin["taux"])
    r = {"trace": trace, "lignes": n_lignes, "jetons": len(jetons), "couches": len(couches),
         "capacite": capacite, "experts": nb_experts, "fraction_capacite": round(capacite / nb_experts, 3),
         "h_pin": round(h, 4),
         "h_pin_par_couche": {int(c): round(float(v), 3) for c, v in pin["taux_par_couche"].items()},
         "h_lru": round(float(lru["taux"]) if isinstance(lru, dict) and "taux" in lru else float(lru), 4)}
    r.update(verdict(h, capacite, nb_experts))
    return r


def verdict(h: float, capacite: int, nb_experts: int) -> dict:
    uniforme = capacite / nb_experts
    if h < SEUIL_REFUTE:
        v = "RÉFUTÉ — le cache n engage pas (S3 ≈ S2 : les froids restent des froids)"
    elif abs(h - PREDIT) <= TOLERANCE:
        v = "TENU"
    else:
        v = "TENU, mieux que prédit : bornes S1/S3 à recalculer à la valeur"
    if h <= uniforme + 0.02:
        v += f" ; h ≈ routage uniforme ({uniforme:.2f}) : aucune concentration à exploiter"
    return {"predit": PREDIT, "tolerance": TOLERANCE, "seuil_refute": SEUIL_REFUTE,
            "h_uniforme": round(uniforme, 3), "verdict": v}


def prendre_trace(model: str, sortie: str, jetons_min: int, max_tokens: int, max_model_len: int) -> int:
    """Même prise que `outils/trace-taux-succes-experts.py` (invites réelles,
    b=1, graphes off) bornée en jetons : les INVITES sont importées de là."""
    os.environ["ACVRAM_TRACE_ROUTAGE"] = sortie
    os.environ.setdefault("ACVRAM_DISABLE_CUDA_GRAPHS", "1")
    ici = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("trace_tse", os.path.join(ici, "../../trace-taux-succes-experts.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.memory import trace_routage
    from acvram.server.chat import load_tokenizer
    tok = load_tokenizer(model)
    if tok is None:
        sys.exit("ÉCHEC / CAUSE : pas de tokenizer.json / SUITE : --model")
    t0 = time.time()
    loaded = load_model(model, dtype=torch.bfloat16, max_model_len=max_model_len)
    engine = Engine(loaded, tok, max_batch_size=1, max_model_len=max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    print(f"[c9-m-trace] chargé en {time.time() - t0:.0f} s, régime {engine.regime_ligne()}", flush=True)
    total, n = 0, 0
    while total < jetons_min or n < len(mod.INVITES):
        invite = mod.INVITES[n % len(mod.INVITES)]
        if n >= len(mod.INVITES):
            invite = f"# variante {n // len(mod.INVITES)}\n{invite}"
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": invite}], add_generation_prompt=True),
                         add_special_tokens=False)
        total += len(list(engine.generate(ids, params)))
        n += 1
    trace_routage.fermer()
    print(f"[c9-m-trace] {n} requêtes, {total} jetons → {sortie}", flush=True)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--trace")
    ap.add_argument("--sortie", default="trace-c9-m-trace.txt")
    ap.add_argument("--jetons", type=int, default=5000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--capacite", type=int, default=52)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not a.trace and not a.model:
        ap.error("--model (prise + analyse) ou --trace (analyse seule)")
    trace = a.trace
    if not trace:
        prendre_trace(a.model, a.sortie, a.jetons, a.max_tokens, a.max_model_len)
        trace = a.sortie
    r = analyser(trace, a.capacite, a.experts)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    pc = r["h_pin_par_couche"]
    print(f"[c9-m-trace] {r['jetons']} jetons, {r['couches']} couches, C = {a.capacite}/{a.experts} "
          f"({r['fraction_capacite']:.0%}) ; prédit h_pin {PREDIT} ± {TOLERANCE}, réfuté < {SEUIL_REFUTE}")
    print(f"  mesuré : h_pin {r['h_pin']:.3f} (couches min {min(pc.values()):.2f} · max {max(pc.values()):.2f}) · "
          f"h_lru {r['h_lru']:.3f} · uniforme {r['h_uniforme']:.2f}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
