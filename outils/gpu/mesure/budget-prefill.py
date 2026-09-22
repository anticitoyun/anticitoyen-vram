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
    lot      N invites préfillées ENSEMBLE. Un coût fixe par passe — s'il
             existe — est alors partagé par les N, donc le surcoût par jeton
             doit être divisé par N. C'est le régime de charge réelle, où
             les prefills arrivent groupés, et il peut PRÉCISER le verdict
             rendu en `seule` sans le contredire.

GARDE-FOU POSÉ AVANT LA MESURE : si le prix relevé en `seule` dépasse ce que
`charge` rachète, **le défaut reste 0** et on l'écrit. Un budget est
exactement le genre de réglage qui a l'air raisonnable et qui coûte 30 %.

UNE VALEUR PAR PROCESSUS. Balayer plusieurs budgets dans un même processus
fragmenterait la carte pour les suivants : le classement trouvé serait
l'artefact du balayage.

LE REGIME APPARTIENT AU RESULTAT
-------------------------------
La manche se court avec ACVRAM_MAX_GRAPHS=64 dans LES DEUX bras, pour
retirer du bruit des deux cotes a la fois. Le resultat s appelle donc
« gain du budget de jetons, A 64 PLACES DE GRAPHES » — et il NE SE
TRANSPORTE PAS au produit, dont le defaut est 16.

Ce n est pas une precaution de style : a 16 places, un prefill decoupe
rencontre PLUS de formes (plus de valeurs de `b` et de `nblk`), donc met
plus de pression sur un cache deja sature — sans eviction. Le decoupage
peut etre gagnant a 64 et perdant a 16, par un mecanisme qui n a rien a
voir avec le budget.

**SI LE DEFAUT DU PRODUIT RESTE A 16, CETTE MANCHE EST A REFAIRE A 16
AVANT DE POSER UNE VALEUR DE BUDGET PAR DEFAUT.** Ecrit d avance pour
n avoir pas a trancher plus tard entre un chiffre commode et un chiffre
applicable.

La colonne `places_graphes` porte la valeur REELLEMENT vue par le module,
pas celle qu on croit avoir posee : `MAX_GRAPHS` est lu A L IMPORT, et une
variable posee apres ne fait rien, EN SILENCE.

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
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

A = _RACINE


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
    ap.add_argument("--regime", choices=("seule", "charge", "lot"), required=True)
    ap.add_argument("--lot", type=int, default=12,
                    help="nombre d invites prefillees ensemble, regime `lot`")
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

    n_lot = {"charge": ns.concurrence + 1, "lot": ns.lot}.get(ns.regime, 1)
    if ns.processeur:
        charge = load_model(ns.processeur, dtype=torch.float32,
                            device_override="cpu",
                            max_model_len=ns.max_model_len,
                            max_concurrent_seqs=n_lot)
    else:
        charge = load_model(os.path.join(A, ns.modele),
                            max_model_len=ns.max_model_len,
                            max_concurrent_seqs=n_lot)
    # `_replanifier` (loader.py) n'agit que si `detect_rig()` voit un GPU —
    # au rodage `--processeur`, le kwarg ci-dessus est ignoré. Le porter
    # explicitement au lot réel plutôt que de laisser le défaut du manifeste
    # (sage-reprise-ordre-18-09 §Suite).
    charge.plan.kv_planned_seqs = n_lot
    moteur = Engine(charge, None, max_batch_size=n_lot,
                    max_model_len=ns.max_model_len)

    # CONTRÔLE : la variable est-elle seulement lue ? Une consigne posée qui
    # ne va nulle part est un balayage muet — c'est arrivé le 9/09 avec
    # MAXTOK=65536. On interroge le moteur, pas l'environnement.
    lu = moteur._budget_jetons()
    if lu != ns.budget:
        raise SystemExit(f"budget non lu par le moteur : {lu} != {ns.budget}")
    # Meme controle pour les places de graphes, mais il ne peut PAS se faire
    # apres coup : lu a l import, il faut interroger le module et non
    # l environnement. Un bras qui differe de l autre prouverait la prise
    # a posteriori — une preuve n est pas un controle.
    from acvram.engine import graphs as _graphes
    places = _graphes.MAX_GRAPHS
    voulu = os.environ.get("ACVRAM_MAX_GRAPHS")
    if voulu is not None and int(voulu) != places:
        raise SystemExit(f"places de graphes non prises : {places} != {voulu}")

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

    p8 = SamplingParams(temperature=0.0, max_tokens=8)
    if ns.regime == "lot":
        # Toutes admises au meme pas : c est ce qui fait partager le cout
        # fixe. Des invites DIFFERENTES entre elles, sinon le cache de
        # prefixe servirait les suivantes et il n y aurait qu un prefill.
        groupe = [moteur.add_request(_invite(ns.invite, 101 + 13 * i, vocab),
                                     p8, request_id=f"L{i}")
                  for i in range(ns.lot)]
        longue = groupe[-1]
    else:
        longue = moteur.add_request(_invite(ns.invite, 11, vocab), p8,
                                    request_id="L")
        groupe = [longue]
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
        while not all(s_.prefilled for s_ in groupe):
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
    # `jetons_prefilles` : ce qui a REELLEMENT ete passe en avant, pas ce
    # qui a ete demande. En `lot` il vaut N x invite, et l energie par jeton
    # ne se compare qu a travers lui.
    prefilles = sum(len(s_.prompt_ids) for s_ in groupe)
    print(f"{ns.budget}\t{ns.regime}\t{places}\t{prefilles}\t"
          f"{latence*1000:.1f}\t{passes}\t{produits}\t{produits/latence:.2f}\t"
          f"{j:.1f}\t{par_kj:.1f}\t{r['watts']:.0f}\t{r['temp_max']}\t"
          f"{r['bridages']}\t{r['invalidations']}")
    print(f"[fini] budget={ns.budget} regime={ns.regime} "
          f"places={places} latence_prefill={latence*1000:.0f} ms passes={passes} "
          f"voisines={produits} jetons", file=sys.stderr)
    return 0


if __name__ == "__main__":
    print("budget\tregime\tplaces_graphes\tjetons_prefilles\tlatence_prefill_ms\tpasses_prefill\t"
          "jetons_voisines\tjetons_s_voisines\tjoules\tjetons_par_kJ\twatts\t"
          "temp_max\tbridages\tinvalidations", file=sys.stderr)
    sys.exit(main(sys.argv))
