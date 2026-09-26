#!/usr/bin/env python3
"""À quelle concurrence la capture de graphes échoue, et POURQUOI.

Le moteur échoue à douze séquences avec les graphes actifs — configuration par
défaut — et passe à huit. Deux causes possibles, qui ne se corrigent pas du
même côté :

    la PLACE manque          la marge de VRAM ne depend pas de max_batch_size,
                             et chaque graphe retient sa memoire d activations
    une operation INTERDITE  une allocation ou une synchronisation dans la
                             capture, qu un certain nombre de sequences declenche

**Ce script rend UNE ligne pour UNE valeur, et doit être appelé une fois par
valeur, dans un processus neuf.** Balayer 8, 9, 10, 11, 12 dans un seul
processus fragmenterait la mémoire pour l'essai suivant : le seuil trouvé
serait alors l'artefact du balayage lui-même, pas une propriété du moteur.

LE CRITERE QUI TRANCHE, fixé avant la mesure : si le seuil se DEPLACE selon la
mémoire libre au départ, la cause est l'allocation ; s'il reste au même rang
quelle que soit la fragmentation, elle est structurelle. C'est cette invariance
qui ferait la démonstration — pas un échec isolé sur une carte encombrée.

D'où la colonne `libre_depart` : un seuil ne vaut rien sans la mémoire libre à
laquelle il se rapporte.
"""
import argparse
import gc
import os
import sys

import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


A = _RACINE


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modele", default="Qwen3-4B-srcgguf-nvfp4")
    ap.add_argument("--seqs", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--invite", type=int, default=256)
    ap.add_argument("--sans-graphes", action="store_true")
    ns = ap.parse_args(argv[1:])

    if torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES=0 obligatoire")

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    g = 2**30
    libre0, total = torch.cuda.mem_get_info(0)
    charge = load_model(os.path.join(A, ns.modele), max_model_len=ns.max_model_len)
    moteur = Engine(charge, None, max_batch_size=ns.seqs,
                    max_model_len=ns.max_model_len,
                    enable_cuda_graphs=not ns.sans_graphes)
    torch.cuda.synchronize()
    libre_apres_chargement, _ = torch.cuda.mem_get_info(0)

    # CONTROLE OBLIGATOIRE, ajoute le 10/09 apres la mesure de poste3 : un plan
    # qui exile un seul MLP en RAM hote desactive TOUS les graphes CUDA
    # (`graphs.py::_eligible`, « poids en flux depuis la RAM hote »). La manche
    # rendrait alors `captures = 0` pour une raison qui n a rien a voir avec ce
    # qu on mesure — et ressemblerait a une refutation du correctif. On releve
    # donc l exil et la raison d eligibilite A COTE du verdict, et une manche
    # qui exile ne dit rien.
    from acvram.engine.layers import QuantLinear
    exiles = sum(1 for m_ in charge.model.modules()
                 if isinstance(m_, QuantLinear) and m_.streamed is not None)
    g_ = getattr(moteur, "graphs", None)
    actifs = bool(getattr(g_, "enabled", False))
    raison = (getattr(g_, "raison", "") or "").replace("\t", " ")[:80]

    params = SamplingParams(temperature=0.0, max_tokens=8)
    for i in range(ns.seqs):
        ids = [(i * 7919 + j * 31 + 11) % 30000 + 1 for j in range(ns.invite)]
        moteur.add_request(ids, params, request_id=f"r{i}")

    # La marge AU MOMENT DE LA CAPTURE est le chiffre qui distingue les deux
    # causes : confortable et pourtant en echec, le mecanisme « la place
    # manque » est faux malgre sa beaute.
    libre_avant_pas, _ = torch.cuda.mem_get_info(0)
    verdict, detail = "OK", ""
    # Le COMPTE DE GRAPHES separe deux mecanismes qui predisent le meme
    # symptome : si la memoire libre est confortable et que les graphes sont
    # nombreux, ce n est pas la marge qui manque mais la cle du graphe qui en
    # fabrique trop — `nblk` y figure, si bien qu une croissance du contexte
    # RECAPTURE au lieu de patcher. Si la memoire est mince avec peu de
    # graphes, c est la marge. Si les deux sont serres, c est le meme
    # probleme vu de deux cotes : beaucoup de graphes FONT que la marge manque.
    def _etat_graphes():
        g_ = getattr(moteur, "graphs", None)
        if g_ is None:
            return 0, 0
        return getattr(g_, "captures", 0), len(getattr(g_, "graphs", {}) or {})
    try:
        for _ in range(12):
            moteur.step()
            if not moteur.running and not moteur.waiting:
                break
    except Exception as exc:                                # noqa: BLE001
        # Le NOM de l exception ne dit pas quelle operation est interdite : le
        # message, si. Le retenir a coute une manche le 10/09.
        import traceback
        traceback.print_exc()
        verdict = "ECHEC"
        detail = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    libre_fin, _ = torch.cuda.mem_get_info(0)
    captures, vivants = _etat_graphes()

    print(f"{ns.seqs}\t{'sans' if ns.sans_graphes else 'avec'}\t"
          f"{libre0/g:.2f}\t{libre_apres_chargement/g:.2f}\t"
          f"{libre_avant_pas/g:.2f}\t{libre_fin/g:.2f}\t"
          f"{captures}\t{vivants}\t{exiles}\t{'oui' if actifs else 'non'}\t"
          f"{verdict}\t{detail}\t{raison}")
    del moteur, charge
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    print("seqs\tgraphes\tlibre_depart\tapres_chargt\tavant_pas\tfin\tcaptures"
          "\tvivants\texiles\tactifs\tverdict\tdetail\traison",
          file=sys.stderr)
    sys.exit(main(sys.argv))
