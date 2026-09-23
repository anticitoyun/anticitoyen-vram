#!/usr/bin/env python3
"""Pièce 48 : lit les CSV de `ncu-etroites-p48.sh` et range chaque bras dans
UNE des quatre issues écrites avant la mesure (I1 dépaquetage, I2 latence,
I3 occupation, I4 rien à gagner) — ou « indécis », qui est un résultat.

Le classement est celui du verdict, recopié ici pour qu'aucune lecture ne se
négocie après coup : c'est le programme qui tranche, pas le lecteur."""
import csv
import glob
import os
import sys
from collections import defaultdict

CLES = {
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "emission",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "warps",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "long_sb",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "short_sb",
    "dram__bytes.sum": "octets",
    "gpu__time_duration.sum": "duree_ns",
}


def issue(m: dict) -> str:
    e, d, w, l = m.get("emission"), m.get("dram"), m.get("warps"), m.get("long_sb")
    if None in (e, d):
        return "indécis (métriques manquantes)"
    if d >= 80:
        return "I4 — rien à gagner (bande saturée)"
    if e >= 70:
        return "I1 — dépaquetage, émission saturée"
    if e < 50 and (l or 0) >= 40:
        return "I2 — latence mal couverte"
    if (w or 100) < 30:
        return "I3 — occupation / vagues partielles"
    return f"indécis (émission {e:.0f} %, dram {d:.0f} %, long_sb {l or 0:.0f} %)"


def lire(chemin: str) -> dict:
    par_noyau: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(chemin, newline="") as f:
        for ligne in csv.DictReader(l for l in f if l.startswith('"') or "," in l):
            nom = ligne.get("Kernel Name") or ligne.get("Kernel") or "?"
            cle = CLES.get((ligne.get("Metric Name") or "").strip())
            if not cle:
                continue
            try:
                par_noyau[nom[:48]][cle].append(float((ligne.get("Metric Value") or "0").replace(",", "")))
            except ValueError:
                pass
    return {n: {k: sum(v) / len(v) for k, v in m.items()} for n, m in par_noyau.items()}


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/poste1-p48-ncu"
    for chemin in sorted(glob.glob(os.path.join(d, "*.csv"))):
        bras = os.path.basename(chemin)[:-4]
        noyaux = lire(chemin)
        if not noyaux:
            print(f"{bras} : AUCUNE métrique lue (voir {bras}.log)")
            continue
        for nom, m in noyaux.items():
            print(f"{bras} · {nom}")
            print("   " + "  ".join(f"{k}={v:.1f}" for k, v in sorted(m.items())))
            print(f"   → {issue(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
