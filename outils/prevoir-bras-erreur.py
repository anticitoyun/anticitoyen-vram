#!/usr/bin/env python3
"""Prediction SCELLEE du bras `erreur`, avant que quiconque le lance.

Simule le glouton avec chacune des cles sur les donnees deja collectees — les
deux SNR et le cout de chaque tenseur, reconstruits depuis cinq manifestes a
budgets croissants. Aucune carte.

Ce qui est reconstruit, et le TEMOIN qui le valide : la cle `snr` simulee doit
reproduire les 198 promus du bras A reellement converti. Si elle ne les
reproduit pas, la simulation ne vaut rien et la prediction est retiree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
DOS = [(0.0, "Llama-2-7b-nvfp4"), (4.50, "Llama-2-7b-quota-4g50"),
       (5.00, "Llama-2-7b-quota-5g00"), (5.50, "Llama-2-7b-quota-5g50"),
       (6.00, "Llama-2-7b-quota-6g00"), (6.55, "Llama-2-7b-quota-plafond")]
GIO = 1024 ** 3
BUDGET = 6.00
# mesures des bras deja convertis
A_PROMUS, A_PPL = 198, 5.4918
B_PROMUS, B_PPL = 149, 5.4482
PLAFOND_PPL = 5.4144
ETALON = 5.4141


def octets(v: dict) -> int:
    out, inn = v["shape"][0], v["shape"][1]
    g = v.get("group_size") or 128
    ng = max(1, inn // g)
    if v["format"] == "int8":
        return out * inn + out * ng * 2 + out * ng
    if v["format"] == "nvfp4":
        return out * (inn // 2) + out * (inn // 16)
    return int(out * inn * v.get("bpw", 16.0) / 8)


err = lambda s: 10.0 ** (-s / 20.0)


def main() -> int:
    et = {}
    for b, nom in DOS:
        p = BASE / nom / "acvram_manifest.json"
        if not p.exists():
            print(f"ECHEC / CAUSE: {nom} absent")
            return 2
        et[b] = json.load(open(p))
    bs = sorted(et)
    tens = {b: et[b]["tensors"] for b in bs}
    plancher = (et[6.00]["budget"] or {}).get("plancher_gib")
    if plancher is None:
        print("ECHEC / CAUSE: le manifeste a 6,00 Gio ne porte pas son plancher")
        return 2
    cand = []
    for k in sorted(tens[bs[0]]):
        base = None
        for b in bs:
            v = tens[b].get(k)
            if v is None:
                break
            if v["format"] == "nvfp4":
                base = v
            elif v["format"] == "int8" and base:
                cand.append({"nom": k, "sb": base["out_snr_db"],
                             "sp": v["out_snr_db"],
                             "cout": max(octets(v) - octets(base), 1)})
                break
    reste_total = BUDGET * GIO - plancher * GIO
    print(f"{len(cand)} candidats, plancher {plancher} Gio, "
          f"budget {BUDGET} Gio, reste a depenser "
          f"{reste_total / 2**20:.0f} Mio\n")

    cles = {
        "snr":     lambda c: -( (c["sp"] - c["sb"]) / c["cout"] ),
        "inverse": lambda c: +( (c["sp"] - c["sb"]) / c["cout"] ),
        "erreur":  lambda c: -( (err(c["sb"]) - err(c["sp"])) / c["cout"] ),
        # QUATRIEME BRAS, nomme d'avance par chef : si `erreur` marche
        # seulement parce qu'il promeut MOINS et les MAL QUANTIFIES, alors
        # trier par SNR de base croissant en ignorant le cout doit faire
        # aussi bien. Ce bras separe la transformation de son effet de bord.
        "base_croissant": lambda c: c["sb"],
    }
    resultats = {}
    for nom, cle in cles.items():
        reste, promus = reste_total, []
        for c in sorted(cand, key=cle):
            if c["cout"] <= reste:
                promus.append(c["nom"])
                reste -= c["cout"]
        resultats[nom] = promus
        print(f"  {nom:16s} {len(promus):3d} promus, "
              f"{reste / 2**20:7.1f} Mio inemployes")

    # ---- TEMOIN : la simulation doit reproduire le bras A reellement converti
    ecart_a = abs(len(resultats["snr"]) - A_PROMUS)
    print(f"\nTEMOIN — la cle `snr` simulee doit reproduire les {A_PROMUS} "
          f"promus du bras A converti")
    print(f"  simule {len(resultats['snr'])}, ecart {ecart_a}")
    if ecart_a > 6:
        print("  SIMULATION REFUSEE : l'ecart depasse la discretisation en "
              "cinq budgets.\n  La prediction est RETIREE — mes couts ou mes "
              "SNR reconstruits sont faux.")
        return 3
    print("  reproduite a la discretisation pres : la simulation tient\n")

    def desaccord(a, b, k):
        return len(set(a[:k]) & set(b[:k]))
    e = resultats["erreur"]
    for autre in ("snr", "inverse", "base_croissant"):
        o = resultats[autre]
        n = min(len(e), len(o))
        print(f"  erreur contre {autre:16s} "
              f"communs top-10 {desaccord(e, o, 10)}/10, "
              f"top-100 {desaccord(e, o, min(100, n))}/{min(100, n)}, "
              f"ensembles {len(set(e) & set(o))}/{max(len(e), len(o))}")

    # SECOND TEMOIN, NON PREVU : la cle `inverse` simulee rend le meme compte
    # que le bras B reellement converti. Deux comptes reproduits sur deux
    # ordres tres differents — la reconstruction ne tient pas par hasard.
    ecart_b = abs(len(resultats["inverse"]) - B_PROMUS)
    print(f"\nSECOND TEMOIN, non prevu — la cle `inverse` simulee contre les "
          f"{B_PROMUS} promus du bras B")
    print(f"  simule {len(resultats['inverse'])}, ecart {ecart_b}"
          + ("  reproduite aussi" if ecart_b <= 6 else "  NON reproduite"))

    # PRE-REFUTATION D'UNE EXPLICATION CONCURRENTE, disponible AVANT la mesure.
    n_e, n_i = len(resultats["erreur"]), len(resultats["inverse"])
    print(f"""
PRE-REFUTATION DEJA ACQUISE, sans mesurer : `erreur` promeut {n_e} tenseurs,
  `inverse` en promeut {n_i}. Si le bras `erreur` ameliore l'actuel EN PROMOUVANT
  LE MEME NOMBRE que lui ({A_PROMUS}), l'explication « ca marche parce qu'on
  promeut moins » tombe d'elle-meme. Le confondant qui reste est « on promeut
  les mal quantifies », et c'est `base_croissant` ({len(resultats['base_croissant'])} promus) qui le teste.""")

    print(f"""
=====================  PREDICTION SCELLEE  =====================
  bras A  ordre snr        {A_PROMUS} promus   PPL {A_PPL}
  bras B  ordre inverse    {B_PROMUS} promus   PPL {B_PPL}
  plafond tout promu       225 promus   PPL {PLAFOND_PPL}

  bras `erreur` prevu      {len(resultats['erreur'])} promus

  FOURCHETTE ATTENDUE : {PLAFOND_PPL} <= PPL(erreur) <= {A_PPL}
  ISSUE ESPEREE        : PPL(erreur) < {B_PPL} — la cle fondee fait
                         mieux que l'instrument, et la substitution
                         devient le defaut.
  ISSUE NEUTRE         : {B_PPL} <= PPL(erreur) < {A_PPL} — elle ameliore
                         l'actuel sans battre l'inverse ; il reste un
                         ordre a chercher.
  ISSUE QUI ME GENE    : PPL(erreur) >= {B_PPL} ET le bras
                         `base_croissant` fait aussi bien ou mieux.
                         Cela dirait que ce qui marche n'est PAS ma
                         transformation mais « promouvoir moins, et les
                         mal quantifies » — un effet de bord, pas un
                         critere. Le quatrieme bras est la pour le
                         separer, et il est nomme AVANT la mesure.
  ISSUE QUI REFUTE     : PPL(erreur) > {A_PPL} — la cle est pire que
                         l'actuelle et l'argument tombe.

  CE QUE CETTE PREDICTION N'EST PAS : l'inverse n'a pas donne une BORNE.
  Un ordre inverse est un ordre parmi 225!, choisi parce qu'il est facile
  a nommer. Il etablit qu'AU MOINS 56,3 % du retard au plafond etaient
  recuperables par le seul ordre — un PLANCHER sur le gain accessible,
  pas un plafond. Rien n'interdit qu'un meilleur ordre en recupere 80 %.
================================================================""")
    sorties() / "prediction-bras-erreur.json".write_text(json.dumps(
             {"promus_prevus": {k: len(v) for k, v in resultats.items()},
              "temoin_ecart_bras_A": ecart_a,
              "fourchette": [PLAFOND_PPL, A_PPL],
              "issue_esperee": f"< {B_PPL}",
              "issue_qui_gene": "PPL >= B et base_croissant aussi bon",
              "communs_erreur_inverse": len(set(resultats["erreur"])
                                            & set(resultats["inverse"]))},
             indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
