#!/usr/bin/env python3
"""M1 (Sage, revue/sage-cache-experts-13-09.md §5) : taux de succès d'experts,
sans rien changer au moteur — `trace_routage.taux_de_succes` existe depuis un
moment et n'a jamais tourné.

Invites RÉELLES (pas des identifiants aléatoires comme `bench_decode` : un
routage sur du bruit ne dit rien de la concentration d'un usage réel), une
requête par invite, `ACVRAM_TRACE_ROUTAGE` posé et `ACVRAM_DISABLE_CUDA_GRAPHS=1`
(la trace synchronise — `trace_routage.py:99`, `sage-cache-experts-13-09.md §7).

Usage :
    outils/carte.sh .venv/bin/python outils/trace-taux-succes-experts.py \\
        --model /chemin/Qwen3-Coder-30B-A3B-nvfp4 \\
        --sortie trace-Qwen3-Coder-30B-decode-b1.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Le nom porte le régime (Sage §7) : une trace à b=1 ne prédit pas b=12.
DEFAUT_SORTIE = "trace-Qwen3-Coder-30B-decode-b1.txt"

# Vingt invites de code RÉELLES, en français et anglais comme le corpus
# d'usage réel, de familles différentes (fonction, bogue, refactor, format,
# algorithme, test) pour ne pas biaiser le routage vers un seul style.
INVITES = [
    "Write a Python function that merges two sorted lists into one sorted list "
    "without using the built-in sort.",
    "Corrige ce code : `def moyenne(l): return sum(l)/len(l)` — il plante sur "
    "une liste vide, ajoute un cas particulier.",
    "Refactor this function to use a dict comprehension instead of a for loop:\n"
    "```python\nresult = {}\nfor k, v in items:\n    result[k] = v * 2\n```",
    "Écris une fonction qui vérifie si une chaîne est un palindrome, en "
    "ignorant la casse et les espaces.",
    "Implement a binary search function in Python that returns the index of "
    "the target, or -1 if not found.",
    "Explique en une phrase la différence entre une liste et un tuple en "
    "Python, puis donne un exemple de chacun.",
    "Write a JSON schema for a user object with fields: id (integer), name "
    "(string), email (string), active (boolean).",
    "Convertis cette boucle for en list comprehension : "
    "`out = []\\nfor x in data:\\n    if x > 0:\\n        out.append(x*2)`",
    "Write a SQL query that selects the top 5 customers by total order amount "
    "from tables `customers` and `orders`.",
    "Écris un test unitaire pytest pour une fonction `diviser(a, b)` qui lève "
    "ZeroDivisionError si b vaut 0.",
    "Implement a simple LRU cache in Python using collections.OrderedDict.",
    "Corrige ce bogue : une fonction récursive de calcul de factorielle "
    "boucle infiniment pour n=0.",
    "Write a bash one-liner that finds all .py files modified in the last 24 "
    "hours under the current directory.",
    "Explique pourquoi cette fonction async ne fonctionne pas comme prévu et "
    "propose un correctif : `async def f(): time.sleep(1)`.",
    "Implement quicksort in Python, in-place, with the last element as pivot.",
    "Écris une classe Python `Point` avec x, y, une méthode `distance` vers un "
    "autre point, et `__repr__`.",
    "Write a regular expression that matches valid IPv4 addresses.",
    "Convertis ce code Python 2 en Python 3 : "
    "`print 'hello', name\\nfor k, v in d.iteritems(): print k, v`",
    "Implement a function that flattens an arbitrarily nested list in Python.",
    "Écris une fonction qui compte les occurrences de chaque mot dans un "
    "texte, insensible à la casse, ponctuation retirée.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="dossier du modèle converti")
    ap.add_argument("--sortie", default=DEFAUT_SORTIE,
                    help=f"chemin de la trace (défaut : {DEFAUT_SORTIE})")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="jetons max par requête (défaut 1024)")
    ap.add_argument("--min-jetons-total", type=int, default=50_000,
                    help="seuil bas (Sage §5) : répète les invites si atteint pas")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    # Doit être posé AVANT tout import qui touche trace_routage (le fichier de
    # sortie s'ouvre à la première écriture, mais _ACTIF se fige à l'import).
    os.environ["ACVRAM_TRACE_ROUTAGE"] = args.sortie
    os.environ.setdefault("ACVRAM_DISABLE_CUDA_GRAPHS", "1")

    import torch

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.server.chat import load_tokenizer

    tokenizer = load_tokenizer(args.model)
    if tokenizer is None:
        print("ÉCHEC / CAUSE : pas de tokenizer.json dans le dossier du modèle "
              "/ SUITE : vérifier --model", file=sys.stderr)
        return 2

    t0 = time.time()
    loaded = load_model(args.model, dtype=torch.bfloat16,
                        max_model_len=args.max_model_len)
    print(f"[trace] modèle chargé en {time.time() - t0:.1f} s", file=sys.stderr)

    replan = getattr(loaded.plan, "replanifie_cartes", None)
    if replan and not os.environ.get("ACVRAM_BANC_ACCEPTE_REPLAN"):
        print(f"ÉCHEC / CAUSE : plan rejoué, cartes du manifeste {replan[0]} "
              f"≠ machine {replan[1]} / SUITE : épingler CUDA_VISIBLE_DEVICES "
              f"ou ACVRAM_BANC_ACCEPTE_REPLAN=1", file=sys.stderr)
        return 2

    engine = Engine(loaded, tokenizer, max_batch_size=1,
                    max_model_len=args.max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    total_jetons = 0
    n_requetes = 0
    tour = 0
    while total_jetons < args.min_jetons_total or n_requetes < 20:
        invite = INVITES[n_requetes % len(INVITES)]
        # Une invite legerement variee a chaque tour au-dela du premier passage
        # sur les vingt : repeter le texte EXACT ferait servir le cache de
        # prefixe des le second tour (meme piege que bench_decode), ce qui
        # fausserait a la fois le debit et — ici plus grave — le ROUTAGE, puisque
        # le cache de prefixe court-circuite le prefill dont la trace depend.
        if tour > 0:
            invite = f"# variante {tour}\n{invite}"
        prompt_ids = tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": invite}],
                add_generation_prompt=True),
            add_special_tokens=False)
        t0 = time.time()
        produits = list(engine.generate(prompt_ids, params))
        dt = time.time() - t0
        n = len(produits)
        total_jetons += n
        n_requetes += 1
        print(f"[trace] requête {n_requetes:3d} (tour {tour}) : {n:4d} jetons "
              f"en {dt:5.1f} s — total {total_jetons}/{args.min_jetons_total}",
              file=sys.stderr)
        if n_requetes % len(INVITES) == 0:
            tour += 1

    from acvram.memory import trace_routage
    trace_routage.fermer()
    print(f"[trace] {n_requetes} requêtes, {total_jetons} jetons décodés -> "
          f"{args.sortie}", file=sys.stderr)
    print(f"FAIT / TESTÉ: {args.sortie} ({total_jetons} jetons, {n_requetes} "
          f"requêtes) / RESTE: taux_de_succes() a lancer dessus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
