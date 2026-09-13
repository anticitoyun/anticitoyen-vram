#!/usr/bin/env python3
"""Agrège une trace de routage (`trace_routage.py`, format
`<jeton> <couche> <e1,e2,...>` par ligne) en histogramme
`{couche: [compte_par_expert]}` et l'écrit via `expert_usage.sauvegarder`
dans `<modele>/.acvram_usage.json` — le profil qu'AUTOPIN
(`decider_residents`) lit au chargement.

Usage :
    python outils/trace-vers-profil-usage.py TRACE.txt --model DOSSIER_MODELE
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-experts", type=int, default=128)
    args = ap.parse_args()

    from acvram.memory import expert_usage

    compte: dict[int, torch.Tensor] = {}
    n_lignes = 0
    with open(args.trace, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            _jeton, couche_s, experts_s = ligne.split(" ", 2)
            couche = int(couche_s)
            t = compte.get(couche)
            if t is None:
                t = torch.zeros(args.n_experts, dtype=torch.int64)
                compte[couche] = t
            for e in experts_s.split(","):
                t[int(e)] += 1
            n_lignes += 1

    if not compte:
        print("ÉCHEC / CAUSE : trace vide ou illisible / SUITE : vérifier le "
              "chemin et que la trace a bien tourné", file=sys.stderr)
        return 2

    total = sum(int(t.sum()) for t in compte.values())
    chemin = f"{args.model.rstrip('/')}/{expert_usage.NOM_PROFIL}"
    expert_usage.sauvegarder(compte, chemin)
    sous_seuil = [c for c, t in compte.items()
                 if int(t.sum()) < expert_usage.SEUIL_CONFIANCE]
    print(f"FAIT / TESTÉ: {chemin} ({len(compte)} couches, {n_lignes} "
          f"positions, {total} sélections) / RESTE: "
          f"{'aucune couche sous le seuil' if not sous_seuil else f'{len(sous_seuil)} couches sous le seuil de confiance ({sorted(sous_seuil)})'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
