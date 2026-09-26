#!/usr/bin/env python3
"""Compare deux captures de `ppl-narrow-b12-coder30b.py` : taux de
divergences top-1 (argmax du modèle, pas le jeton forcé) sur les mêmes
positions. Seuil scellé (poste7, revue/poste7-narrow-b12-16-09.md) :
taux(B vs A) ≤ 1,2 x taux(témoin) -- réglé sur les 24 576 positions
partagées, règle le s3@3 de poste3 sur le même critère.

Usage : compare-narrow-b12-16-09.py A.json B.json A-eager.json
"""
import json
import sys

A, B, A_EAGER = (json.load(open(p)) for p in sys.argv[1:4])


def taux(x, y):
    n_div, n_tot = 0, 0
    for rid in x["top1"]:
        ca, cb = x["top1"][rid], y["top1"][rid]
        n = min(len(ca), len(cb))
        n_div += sum(1 for j in range(n) if ca[j] != cb[j])
        n_tot += n
    return n_div, n_tot


div_ba, n_ba = taux(A, B)
div_temoin, n_temoin = taux(A, A_EAGER)
taux_ba = div_ba / n_ba
taux_temoin = div_temoin / n_temoin if n_temoin else 0.0
seuil = 1.2 * taux_temoin
verdict = "CONFORME" if taux_ba <= seuil else "REFUTE"

print(f"B vs A     : {div_ba}/{n_ba} = {taux_ba:.6f}")
print(f"temoin A/A-eager : {div_temoin}/{n_temoin} = {taux_temoin:.6f}")
print(f"seuil (1,2 x temoin) : {seuil:.6f}")
print(f"VERDICT {verdict}")
print(f"preuve A={A['preuve']} B={B['preuve']} A-eager={A_EAGER['preuve']}")
