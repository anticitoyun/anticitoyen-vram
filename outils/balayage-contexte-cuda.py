#!/usr/bin/env python3
"""Combien de modeles du parc basculeraient si l'annonce comptait le contexte CUDA.

Mesure du 10/09/2026 : le contexte CUDA d'un processus PyTorch occupe
**0,70 Gio**, constante a +-2 % sur trois modeles dont les poids vont de 3,00 a
15,79 Gio (`hors torch` = 0,69 / 0,70 / 0,72). Aucune ligne de
`acvram/memory/tiering.py` ne le prevoit.

Le rendre honnete a un cout : une annonce plus haute franchit le seuil d'exil
plus tot, et de l'autre cote du seuil il y a une falaise doublement chere —
61 a 70 % de debit, ET l'explosion de l'empreinte, puisque l'exil d'une seule
couche supprime tous les graphes CUDA (mesure du 10/09 : +11,53 Gio au pic sur
un MLA passe en eager).

**Ce balayage se fait AVANT d'ecrire le correctif, pas apres.** Il ne charge
aucun poids : il rejoue le planificateur sur chaque manifeste, a capacite
nominale puis a capacite reduite de 0,70 Gio — ce qui est exactement l'effet
d'une annonce augmentee d'autant. La capacite est FIXEE en dur et non lue sur
la carte : sans cela le resultat dependrait de qui mesure a cote, et deux
balayages ne seraient pas comparables.
"""
import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTEXTE_CUDA = int(0.70 * 2**30)


def _exiles(plan) -> tuple[int, int]:
    """(couches dont le MLP est exile, couches dont l'attention l'est)."""
    mlp = sum(1 for lp in plan.layers if lp.mlp_storage == "cpu")
    attn = sum(1 for lp in plan.layers if lp.attn_storage == "cpu")
    return mlp, attn


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contexte", type=int, default=CONTEXTE_CUDA,
                    help="octets a retrancher (defaut : 0,70 Gio mesure)")
    ap.add_argument("--limite", type=int, default=0)
    ns = ap.parse_args(argv[1:])

    from acvram.engine.config import ModelSpec
    from acvram.hardware.detect import detect_rig
    from acvram.memory.tiering import auto_plan
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parc import get_all_model_paths

    rig = detect_rig()
    if not rig.gpus:
        raise SystemExit("aucun GPU detecte : le plan n'a pas de sens")
    # Capacite NOMINALE, jamais le libre du moment.
    nominal = [g.total_mem for g in rig.gpus]

    chemins = sorted(get_all_model_paths())
    if ns.limite:
        chemins = chemins[:ns.limite]

    print("modele\tpoids_G\tmlp_avant\tmlp_apres\tattn_avant\tattn_apres\tbascule")
    bascules, vus, illisibles = [], 0, 0
    for c in chemins:
        mf = os.path.join(c, "acvram_manifest.json")
        if not os.path.exists(mf):
            continue
        try:
            m = json.load(open(mf))
            # Le manifeste porte plus de champs que `ModelSpec` n'en accepte
            # (`total_params`, `gdn_a_log_negexp`...). Les passer tels quels
            # rendait 114 modeles « illisibles » : un instrument qui refuse
            # tout le parc se lit comme un parc vide.
            champs = {f.name for f in dataclasses.fields(ModelSpec)}
            spec = ModelSpec(**{k: v for k, v in m["model"].items()
                                if k in champs})
        except Exception as exc:                       # noqa: BLE001
            illisibles += 1
            print(f"{os.path.basename(c)}\tILLISIBLE\t{type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        lignes = []
        for retrait in (0, ns.contexte):
            for g, tot in zip(rig.gpus, nominal):
                g.total_mem = max(1, tot - retrait)
            try:
                plan, _ = auto_plan(spec, rig)
            except Exception as exc:                   # noqa: BLE001
                lignes = None
                print(f"{os.path.basename(c)}\tPLAN IMPOSSIBLE\t"
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                break
            lignes.append((_exiles(plan), plan.total_weight_bytes))
        for g, tot in zip(rig.gpus, nominal):
            g.total_mem = tot
        if not lignes:
            continue
        (m0, a0), poids = lignes[0]
        (m1, a1), _ = lignes[1]
        vus += 1
        bascule = "OUI" if (m1 > m0 or a1 > a0) else ""
        if bascule:
            bascules.append((os.path.basename(c), m0, m1, a0, a1))
        print(f"{os.path.basename(c)}\t{poids/2**30:.2f}\t{m0}\t{m1}\t"
              f"{a0}\t{a1}\t{bascule}")

    print(f"\n{vus} modeles planifies, {illisibles} illisibles, "
          f"{len(bascules)} bascule(nt)", file=sys.stderr)
    for nom, m0, m1, a0, a1 in bascules:
        print(f"  {nom} : MLP {m0} -> {m1}, attn {a0} -> {a1}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
