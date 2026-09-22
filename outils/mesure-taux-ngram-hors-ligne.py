#!/usr/bin/env python3
"""Taux d'acceptation n-gram, rejeu hors ligne (Jérôme, 13/09, suite Sage §7).

N'utilise PAS le moteur, PAS de GPU : rejoue `NGramProposer` (l'algorithme
réel, pas une réimplémentation) sur des jetons DÉJÀ PRODUITS par une
génération antérieure. Un pas sans proposition compte pour 1 jeton (convention
du chantier, `revue/chantier-speculation.md` §2).

Entrée attendue (JSON) : une liste d'objets {"invite": str, "famille": str,
"jetons": [int, ...]} — les jetons PRODUITS (pas l'invite elle-même). Ce
fichier n'existe pas encore au 13/09 : la génération qui le produirait
attend un tour de carte (voir revue/protocole-taux-ngram-code-13-09.md §2
étape 0).

Usage :
    python outils/mesure-taux-ngram-hors-ligne.py SORTIES.json [--n 3 4 5] [--k 8]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acvram.engine.speculative import NGramProposer  # noqa: E402


def _seq(ids: list[int]):
    """Un faux `Sequence` : NGramProposer ne lit que `.all_ids` et `.id`."""
    return SimpleNamespace(all_ids=ids, id=id(ids))


def taux_pour_n(jetons_par_invite: list[list[int]], n: int, k: int) -> dict:
    """Rejoue un NGramProposer figé à `min_ngram=max_ngram=n` sur chaque suite.

    À chaque position t (à partir de n+1, il faut au moins un n-gramme
    complet avant de pouvoir en chercher un second), propose sur le préfixe
    jetons[:t] et compare aux jetons RÉELS jetons[t:t+k] — jamais de
    réinterrogation du modèle, c'est tout le sens du rejeu hors ligne.
    """
    longueurs: list[int] = []
    n_positions = 0
    n_propositions_non_vides = 0
    for jetons in jetons_par_invite:
        prop = NGramProposer(min_ngram=n, max_ngram=n, adaptatif=False)
        for t in range(n + 1, len(jetons)):
            n_positions += 1
            proposition = prop.propose(_seq(jetons[:t]), k).tokens
            if not proposition:
                longueurs.append(0)
                continue
            n_propositions_non_vides += 1
            reels = jetons[t:t + len(proposition)]
            longueur = 0
            for a, b in zip(proposition, reels):
                if a != b:
                    break
                longueur += 1
            longueurs.append(longueur)
    if not longueurs:
        return {"n": n, "tokens_par_pas": None, "n_positions": 0}
    tokens_par_pas = 1.0 + statistics.mean(longueurs)
    return {
        "n": n,
        "n_positions": n_positions,
        "n_propositions_non_vides": n_propositions_non_vides,
        "taux_proposition": round(n_propositions_non_vides / n_positions, 4) if n_positions else 0.0,
        "longueur_acceptee_moyenne": round(statistics.mean(longueurs), 4),
        "tokens_par_pas": round(tokens_par_pas, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sorties", help="JSON des jetons produits (voir docstring)")
    ap.add_argument("--n", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    with open(args.sorties, encoding="utf-8") as fh:
        data = json.load(fh)

    par_famille: dict[str, list[list[int]]] = {}
    tous: list[list[int]] = []
    for item in data:
        jetons = item["jetons"]
        tous.append(jetons)
        par_famille.setdefault(item.get("famille", "?"), []).append(jetons)

    print(f"[INFO] {len(tous)} suites, {sum(len(j) for j in tous)} jetons au total\n")

    resultats = {"global": {}, "par_famille": {}}
    for n in args.n:
        r = taux_pour_n(tous, n, args.k)
        resultats["global"][n] = r
        print(f"n={n} (global)  tokens/pas={r['tokens_par_pas']}  "
              f"taux_proposition={r.get('taux_proposition')}  "
              f"positions={r['n_positions']}")

    for famille, suites in par_famille.items():
        resultats["par_famille"][famille] = {}
        for n in args.n:
            r = taux_pour_n(suites, n, args.k)
            resultats["par_famille"][famille][n] = r
        print(f"\n[{famille}]")
        for n in args.n:
            r = resultats["par_famille"][famille][n]
            print(f"  n={n}  tokens/pas={r['tokens_par_pas']}")

    sortie_json = args.sorties.rsplit(".", 1)[0] + "-taux-ngram.json"
    with open(sortie_json, "w", encoding="utf-8") as fh:
        json.dump(resultats, fh, indent=2, ensure_ascii=False)
    print(f"\n[INFO] résultats -> {sortie_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
