#!/usr/bin/env python3
"""PPL appariée par bootstrap sur les fenêtres d'`acvram eval --json` (fonction de la pièce 138 bis, poste4 24/09,
`bootstrap_ppl` : rééchantillonnage des fenêtres, PPL = exp(Σ nll / Σ positions), 20 000 tirages) — ici dans le dépôt,
appariée : le MÊME tirage de fenêtres sert aux deux modèles, d'où un Δ par tirage et son IC95 / z.
Usage : ppl-appariee-bootstrap.py A.json B.json [C.json …] [--tirages 20000] [--graine 0] → tableau : PPL de chacun,
puis pour chaque modèle ≠ A : Δ = PPL/PPL_A − 1 (moyenne bootstrap), IC95, z = Δ / σ. Seuil de conclusion : |z| > 2 (scellé).
Les fenêtres doivent être identiques (même corpus, --window, --stride, --max-tokens) : vérifié par le nombre de positions."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys


def fenetres(chemin: str) -> list[tuple[float, int]]:
    d = json.load(open(chemin))
    r = d[0] if isinstance(d, list) else d
    return [(float(a), int(b)) for a, b in r["par_fenetre"]]


def ppl(paires: list[tuple[float, int]], idx: list[int]) -> float:
    s = sum(paires[i][0] for i in idx)
    n = sum(paires[i][1] for i in idx)
    return math.exp(s / n)


def bootstrap_apparie(ref: list, autres: list[list], tirages: int, graine: int) -> tuple[list[float], list[list[float]]]:
    rng = random.Random(graine)
    n = len(ref)
    ppl_ref, deltas = [], [[] for _ in autres]
    for _ in range(tirages):
        idx = [rng.randrange(n) for _ in range(n)]
        p0 = ppl(ref, idx)
        ppl_ref.append(p0)
        for k, a in enumerate(autres):
            deltas[k].append(ppl(a, idx) / p0 - 1.0)
    return ppl_ref, deltas


def resume(x: list[float]) -> tuple[float, float, float, float]:
    x = sorted(x)
    m = sum(x) / len(x)
    v = sum((t - m) ** 2 for t in x) / (len(x) - 1)
    return m, v ** 0.5, x[int(0.025 * len(x))], x[int(0.975 * len(x))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--tirages", type=int, default=20000)
    ap.add_argument("--graine", type=int, default=0)
    a = ap.parse_args()
    series = [fenetres(c) for c in a.json]
    n_pos = [sum(b for _, b in s) for s in series]
    if len({len(s) for s in series}) != 1 or len(set(n_pos)) != 1:
        print(f"ÉCHEC : fenêtres non appariées (fenêtres {[len(s) for s in series]}, positions {n_pos})"); return 2
    tout = list(range(len(series[0])))
    print(f"# {len(series[0])} fenêtres, {n_pos[0]} positions, {a.tirages} tirages, graine {a.graine}")
    for c, s in zip(a.json, series):
        print(f"{c}: PPL {ppl(s, tout):.4f}")
    ppl_ref, deltas = bootstrap_apparie(series[0], series[1:], a.tirages, a.graine)
    m, sd, lo, hi = resume(ppl_ref)
    print(f"référence {a.json[0]} : PPL bootstrap {m:.4f} ± {sd:.4f} (IC95 [{lo:.4f} ; {hi:.4f}])")
    for c, d in zip(a.json[1:], deltas):
        m, sd, lo, hi = resume(d)
        z = m / sd if sd > 0 else (0.0 if abs(m) < 1e-12 else float("inf"))   # identique au bit : Δ 0, z 0
        verdict = "conclusif" if abs(z) > 2 else "non conclusif"
        print(f"{c} − référence : Δ {m * 100:+.3f} % (IC95 [{lo * 100:+.3f} ; {hi * 100:+.3f}] %), z {z:+.2f} → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
