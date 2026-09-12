#!/usr/bin/env python3
"""Le temoin qui distingue « mieux trie » de « moins contraint ».

Une INVERSION est un couple (tenseur non promu, tenseur promu) ou le NON
PROMU a le pire SNR. Sous une selection positionnelle — le quota sature et
l'ordre de parcours decide a la place du SNR — il y en a necessairement.
Apres un tri correct par rang de SNR, il y en a EXACTEMENT ZERO, par
construction. C'est ce qui rend ce temoin utilisable :

    mieux trie       inversions -> 0     promotions inchangees
    moins contraint  inversions > 0      promotions en hausse

Desserrer le quota ne ramene pas les inversions a zero : la selection reste
positionnelle parmi les candidats restants, elle est seulement moins
tronquee. Un tri correct les annule sans toucher au budget. Temoin propose
par poste1 le 9/09/2026.

DEUX PRECAUTIONS, et la premiere a deja fait une victime.

1. LE SNR SE COMPARE A FORMAT EGAL. `out_snr_db` d'un tenseur promu est
   celui de son format FINAL — l'int8 d'apres la promotion. Le comparer au
   `out_snr_db` d'un tenseur reste en nvfp4 compare deux formats en croyant
   comparer deux merites : 44,4 dB contre 20,5 dB de mediane sur 36 modeles,
   parfaitement regulier, et parfaitement vide de sens. Ce script exige donc
   la cle `snr_db_source` (le SNR d'AVANT, pris dans le format que nomme
   `promoted_from`) et REFUSE de conclure sans elle, au lieu de se rabattre
   sur celle qui est presente.

2. L'UNITE DU TEMOIN SUIT LA POLITIQUE DU TRI. Si la promotion est decidee
   par groupe — clé = SNR du pire membre, tout ou rien — alors un tenseur
   promu au bon SNR dans un groupe necessiteux est LEGITIME, et il produit
   pourtant une inversion au niveau tenseur. Compter dans la mauvaise unite
   sanctionne la politique au lieu du defaut. `--unite groupe` existe pour
   cela.
"""
import argparse
import collections
import json
import os
import re
import sys
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

A = MODELES
RE_GROUPE = re.compile(r"^(model\.layers\.\d+\.(?:self_attn|mlp))\.")


def _snr_avant(entree):
    """Le SNR du tenseur AVANT toute promotion, ou None si indisponible.

    Un tenseur non promu n'a jamais ete promu : son `out_snr_db` EST son
    avant. Un tenseur promu porte `promoted_from` et doit fournir son SNR
    d'avant ; sans cette cle, il n'y a pas de reponse, et rendre `out_snr_db`
    a sa place serait rendre le SNR d'un autre format.

    La cle s'appelle `snr_db_source` et non `before` : elle se lit en regard
    de `promoted_from`, qui nomme le format dans lequel elle a ete prise,
    comme `snr_db` se lit en regard de `format`. `before` disait QUAND, pas
    DANS QUOI — or c'est le format qui manquait le jour ou 44,4 dB ont ete
    compares a 20,5. Aucun manifeste ne portait l'une ni l'autre : il n'y a
    rien a menager.
    """
    if "promoted_from" not in entree:
        return entree.get("out_snr_db", entree.get("snr_db"))
    return entree.get("snr_db_source")


def inversions(dossier, unite="tenseur"):
    """Rend (inversions, promus, epargnes, sans_avant).

    `epargnes` sont les tenseurs qui remplissaient le critere de promotion
    mais que le quota a laisses dehors — c'est parmi eux que se trouvent les
    victimes de l'ordre de parcours.
    """
    m = json.load(open(os.path.join(dossier, "acvram_manifest.json")))
    ts = m["tensors"]
    plancher = (m.get("options") or {}).get("snr_floor")

    promus, autres, sans_avant = [], [], 0
    for nom, e in ts.items():
        av = _snr_avant(e)
        if "promoted_from" in e:
            if av is None:
                sans_avant += 1
                continue
            promus.append((nom, av))
        elif av is not None:
            autres.append((nom, av))

    if unite == "groupe":
        # La cle d'un groupe est le SNR de son PIRE membre : c'est ce que la
        # politique « par groupe » classe, donc c'est ce qui doit etre
        # compare. Un groupe est promu des qu'un de ses membres l'est.
        pire, promu_g = {}, set()
        for nom, av in promus + autres:
            mo = RE_GROUPE.match(nom)
            g = mo.group(1) if mo else nom
            pire[g] = min(av, pire.get(g, av))
        for nom, _ in promus:
            mo = RE_GROUPE.match(nom)
            promu_g.add(mo.group(1) if mo else nom)
        promus = [(g, s) for g, s in pire.items() if g in promu_g]
        autres = [(g, s) for g, s in pire.items() if g not in promu_g]

    # Seuls les EPARGNES comptent : un tenseur au-dessus du plancher n'etait
    # pas candidat, son meilleur SNR n'est pas une injustice.
    if plancher is None:
        epargnes = autres
    else:
        epargnes = [(n, s) for n, s in autres if s < plancher]

    if not promus or not epargnes:
        return 0, len(promus), len(epargnes), sans_avant
    pire_promu = max(s for _, s in promus)
    # Une inversion par couple : chaque epargne compte les promus qu'il aurait
    # du devancer. Le tri rend le compte lineaire au lieu de quadratique.
    snrs_promus = sorted(s for _, s in promus)
    total = 0
    import bisect
    for _, s in epargnes:
        if s < pire_promu:
            total += len(snrs_promus) - bisect.bisect_right(snrs_promus, s)
    return total, len(promus), len(epargnes), sans_avant


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modeles", nargs="*")
    ap.add_argument("--unite", choices=("tenseur", "groupe"), default="tenseur")
    ns = ap.parse_args(argv[1:])

    cibles = ns.modeles or sorted(os.listdir(A))
    muets, lignes = 0, []
    cumul = collections.Counter()
    for nom in cibles:
        d = nom if os.path.isdir(nom) else os.path.join(A, nom)
        if not os.path.isfile(os.path.join(d, "acvram_manifest.json")):
            continue
        try:
            inv, np_, ne, sa = inversions(d, ns.unite)
        except Exception as e:                              # noqa: BLE001
            print(f"  {nom} : illisible ({str(e)[:40]})", file=sys.stderr)
            continue
        if sa:
            muets += 1
            lignes.append((nom, None, np_, ne, sa))
            continue
        if np_ == 0:
            continue
        cumul["inversions"] += inv
        cumul["promus"] += np_
        cumul["epargnes"] += ne
        lignes.append((nom, inv, np_, ne, 0))

    print(f"{'modele':46s} {'inversions':>11s} {'promus':>7s} {'epargnes':>9s}")
    for nom, inv, np_, ne, sa in lignes:
        v = "SANS snr_db_source" if inv is None else f"{inv:11d}"
        print(f"{nom[:46]:46s} {v:>11s} {np_:7d} {ne:9d}")
    if cumul["promus"]:
        print(f"\ncumul  inversions {cumul['inversions']}  promus "
              f"{cumul['promus']}  epargnes {cumul['epargnes']}  "
              f"unite {ns.unite}")
    else:
        # Ne JAMAIS imprimer « inversions 0 » quand aucun modele n'etait
        # mesurable : un zero qui vient de l'absence de donnee se lit
        # exactement comme un zero qui vient d'un tri parfait.
        print("\naucun modele mesurable : pas de cumul.")
    if muets:
        print(f"\n{muets} modeles ne peuvent PAS repondre : leurs tenseurs promus")
        print("ne portent pas le SNR d'avant promotion. Ce n'est pas zero")
        print("inversion, c'est une absence de mesure — l'instrument ne pouvait")
        print("rien rendre d'autre. Reconvertir avec un acvram qui pose la")
        print("cle `snr_db_source` — convert.py l'ecrit au moment de la")
        print("promotion, depuis le `before` que `report.promotions` portait")
        print("deja sans le publier.")


if __name__ == "__main__":
    main(sys.argv)
