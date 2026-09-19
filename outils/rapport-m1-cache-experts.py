#!/usr/bin/env python3
"""C9-M1 (poste7-c9-119b-cache-experts-19-09 § 2) : rapport de taux de succès
d'experts sur une trace RÉELLE (`ACVRAM_TRACE_ROUTAGE=<journal.txt>`, journal
texte par couche de memory/trace_routage.py — pas le .pt de
ACVRAM_TRACE_ROUTAGE_PT) : h_pin(C) et h_lru(C) par couche, apprises/chauffées
sur la première moitié des jetons et JUGÉES sur la seconde, experts distincts
par pas (ce qui se paie en octets à b > 1), et le critère de poste7
Δh = h(E/2) − 0,5 (< 0,10 : pas de cache apprenant ; ≥ 0,25 : cache engagé).
Usage : rapport-m1-cache-experts.py trace.txt [--experts 128] [--capacites 16,32,48,64,96] [--json sortie.json]
À sec, aucune carte. Le nom de la trace porte son régime (b=1 ≠ b=12)."""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.memory.trace_routage import rapport_m1                             # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("trace")
ap.add_argument("--experts", type=int, default=None, help="E du config.json (sinon : plus grand index vu + 1)")
ap.add_argument("--capacites", default="16,32,48,64,96")
ap.add_argument("--entrainement", type=float, default=0.5)
ap.add_argument("--json", default=None)
a = ap.parse_args()
caps = tuple(int(c) for c in a.capacites.split(","))
r = rapport_m1(a.trace, caps, a.entrainement, a.experts)
print(f"trace {os.path.basename(a.trace)} : {r['lignes']} lignes, E={r['experts_vus']}, jugé sur la seconde moitié")
print("  C      h_pin    h_lru")
for c in caps:
    print(f"  {c:3d}   {r['h_pin'][c]:.3f}    {r['h_lru'][c]:.3f}")
d = r["distincts_par_pas"]
if d:
    moy = statistics.fmean(v["moyenne"] for v in d.values())
    lot = statistics.fmean(v["lot_moyen"] for v in d.values())
    rec = statistics.fmean(v["recouvrement"] for v in d.values())
    print(f"  distincts par pas (moyenne des couches) : {moy:.1f} pour un lot moyen de {lot:.1f} ; recouvrement {rec:.2f}")
print(f"  h(E/2={r['capacite_demi']}) : pin {r['h_demi']['pin']:.3f}, lru {r['h_demi']['lru']:.3f} → Δh = {r['delta_h']:+.3f} : {r['verdict_poste7']}")
if a.json:
    json.dump(r, open(a.json, "w"), indent=1, default=str)
    print(f"  json : {a.json}")
