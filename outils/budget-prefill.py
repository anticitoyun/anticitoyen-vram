#!/usr/bin/env python3
"""Ce que coûte et ce que rachète le prefill découpé (ACVRAM_BUDGET_JETONS).

Le découpage RÉDUIT le débit brut d'un prefill : plus de passes, donc plus de
lancements et des lots plus petits. Il ne se justifie que par ce qu'il rend
aux séquences DÉJÀ EN COURS. **Un seul des deux régimes peut donc conclure.**

    seule    une longue invite, personne d'autre. Ne peut que montrer une
             perte : c'est le PRIX du découpage, mesuré seul.
    charge   la même invite pendant que N séquences décodent. Mesure ce que
             le découpage RACHÈTE : les jetons que les N produisent pendant
             que l'invite se précalcule. C'est le régime qui décide.

GARDE-FOU POSÉ AVANT LA MESURE : si le prix relevé en `seule` dépasse ce que
`charge` rachète, **le défaut reste 0** et on l'écrit. Un budget est
exactement le genre de réglage qui a l'air raisonnable et qui coûte 30 %.

UNE VALEUR PAR PROCESSUS. Balayer plusieurs budgets dans un même processus
fragmenterait la carte pour les suivants : le classement trouvé serait
l'artefact du balayage.

CE QUE CE SCRIPT NE MESURE PAS, dit ici plutôt qu'insinué : le nombre de
lancements de noyaux par pas. `passes_prefill` en est le témoin indirect —
il compte les pas qu'a demandés le prefill, pas les noyaux qu'ils ont lancés.
Conclure sur le coût de lancement demanderait un profileur, et `ncu`
surestime les noyaux courts de 45 %.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

A = "/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram"


def _invite(n, graine, vocab):
    # Borne par le vocabulaire REEL : un identifiant hors table indexe hors
    # des embeddings, et le rodage sur un modele minuscule le rencontrerait.
    return [(i * 31 + graine) % max(1, vocab - 1) + 1 for i in range(n)]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modele", default="Qwen3-4B-srcgguf-nvfp4")
    ap.add_argument("--budget", type=int, required=True,
                    help="ACVRAM_BUDGET_JETONS ; 0 = comportement actuel")
    ap.add_argument("--regime", choices=("seule", "charge"), required=True)
    ap.add_argument("--invite", type=int, default=8192)
    ap.add_argument("--concurrence", type=int, default=12,
                    help="sequences deja en decodage, regime `charge`")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--processeur", metavar="DOSSIER",
                    help="rodage de l'instrument sur processeur, sans carte : "
                         "le DOSSIER est un converti minuscule. Les chiffres "
                         "n'ont alors aucune valeur — seul le montage est "
                         "eprouve. « L'instrument avant la mesure. »")
    ns = ap.parse_args(argv[1:])

    import torch
    if not ns.processeur and torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES=0 obligatoire")

    # Posée AVANT l'import du moteur : elle est lue à chaque pas, mais la
    # poser tard laisserait croire à un réglage dynamique qui n'en est pas un.
    os.environ["ACVRAM_BUDGET_JETONS"] = str(ns.budget)

    from energie import Energie
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    if ns.processeur:
        charge = load_model(ns.processeur, dtype=torch.float32,
                            device_override="cpu",
                            max_model_len=ns.max_model_len)
    else:
        charge = load_model(os.path.join(A, ns.modele),
                            max_model_len=ns.max_model_len)
    n_lot = ns.concurrence + 1 if ns.regime == "charge" else 1
    moteur = Engine(charge, None, max_batch_size=n_lot,
                    max_model_len=ns.max_model_len)

    # CONTRÔLE : la variable est-elle seulement lue ? Une consigne posée qui
    # ne va nulle part est un balayage muet — c'est arrivé le 9/09 avec
    # MAXTOK=65536. On interroge le moteur, pas l'environnement.
    lu = moteur._budget_jetons()
    if lu != ns.budget:
        raise SystemExit(f"budget non lu par le moteur : {lu} != {ns.budget}")

    vocab = charge.spec.vocab_size
    court = SamplingParams(temperature=0.0, max_tokens=ns.max_tokens)
    voisines = []
    if ns.regime == "charge":
        for i in range(ns.concurrence):
            voisines.append(moteur.add_request(_invite(64, 7 * i + 3, vocab), court,
                                               request_id=f"v{i}"))
        # On les amène en décodage AVANT de chronométrer : leur propre prefill
        # n'est pas ce qu'on mesure.
        while not all(s.prefilled for s in voisines):
            moteur.step()
        for _ in range(4):
            moteur.step()

    longue = moteur.add_request(_invite(ns.invite, 11, vocab),
                                SamplingParams(temperature=0.0, max_tokens=8),
                                request_id="L")
    produits0 = sum(len(s.output_ids) for s in voisines)
    passes = 0
    class _SansCarte:
        """Rodage sur processeur : aucune energie a relever, et on le dit."""
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def resume(self):
            return {"joules": -1.0, "watts": -1.0, "temp_max": -1,
                    "bridages": "sans_objet", "invalidations": "processeur"}

    if not ns.processeur:
        torch.cuda.synchronize()
    with (_SansCarte() if ns.processeur else Energie(periode=0.2)) as e:
        t0 = time.perf_counter()
        while not longue.prefilled:
            moteur.step()
            passes += 1
        if not ns.processeur:
            torch.cuda.synchronize()
        latence = time.perf_counter() - t0
    produits = sum(len(s.output_ids) for s in voisines) - produits0

    r = e.resume()
    j = r["joules"]
    # Jetons/kJ des VOISINES pendant la fenêtre : c'est le travail utile que
    # le découpage rachète. En régime `seule` il vaut zéro par construction,
    # et la colonne le dit au lieu de laisser croire à une mesure ratée.
    par_kj = (produits / j * 1000) if j > 0 else -1.0
    print(f"{ns.budget}\t{ns.regime}\t{ns.invite}\t"
          f"{latence*1000:.1f}\t{passes}\t{produits}\t{produits/latence:.2f}\t"
          f"{j:.1f}\t{par_kj:.1f}\t{r['watts']:.0f}\t{r['temp_max']}\t"
          f"{r['bridages']}\t{r['invalidations']}")
    print(f"[fini] budget={ns.budget} regime={ns.regime} "
          f"latence_prefill={latence*1000:.0f} ms passes={passes} "
          f"voisines={produits} jetons", file=sys.stderr)
    return 0


if __name__ == "__main__":
    print("budget\tregime\tinvite\tlatence_prefill_ms\tpasses_prefill\t"
          "jetons_voisines\tjetons_s_voisines\tjoules\tjetons_par_kJ\twatts\t"
          "temp_max\tbridages\tinvalidations", file=sys.stderr)
    sys.exit(main(sys.argv))
