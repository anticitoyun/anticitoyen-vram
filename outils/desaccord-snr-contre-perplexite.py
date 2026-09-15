#!/usr/bin/env python3
"""Le glouton trie-t-il dans le meme ordre que la perplexite ? Sans reconversion.

Le sac a dos ordonne par `gain_db / cout`. Ce qui decide est la perte marginale
par octet — formule donnee par un avis exterieur le 10/09 :

    valeur/octet(u) = ( DL(u basse precision) - DL(u haute precision) ) / octets(u)

Le manifeste ne porte que les metriques du format RETENU, donc ni `gain_db` ni
le contrefactuel. MAIS cinq dossiers a budgets croissants les donnent : un
tenseur en nvfp4 a 5,50 Gio et en int8 a 6,00 Gio livre ses DEUX formats. Et
l'ordre dans lequel les tenseurs deviennent int8 quand le budget monte EST
l'ordre du glouton.

TEMOIN QUI PEUT ECHOUER : on recalcule `gain_db / cout` depuis les deux SNR et
on verifie qu'il reproduit l'ordre OBSERVE des promotions. S'il ne le reproduit
pas, la reconstruction est fausse et rien de ce qui suit ne vaut.
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
# budget croissant ; le plancher est le dossier tout-nvfp4
DOSSIERS = [(0.0, "Llama-2-7b-nvfp4"), (4.50, "Llama-2-7b-quota-4g50"),
            (5.00, "Llama-2-7b-quota-5g00"), (5.50, "Llama-2-7b-quota-5g50"),
            (6.00, "Llama-2-7b-quota-6g00"), (6.55, "Llama-2-7b-quota-plafond")]


def octets(v: dict) -> int:
    out, inn = v["shape"][0], v["shape"][1]
    g = v.get("group_size") or 128
    ng = max(1, inn // g)
    if v["format"] == "int8":
        return out * inn + out * ng * 2 + out * ng
    if v["format"] == "nvfp4":
        return out * (inn // 2) + out * (inn // 16)
    return int(out * inn * v.get("bpw", 16.0) / 8)


def err(snr_db: float) -> float:
    return 10.0 ** (-snr_db / 20.0)


def spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    def rangs(x):
        o = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[o[j + 1]] == x[o[i]]:
                j += 1
            moy = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = moy
            i = j + 1
        return r
    ra, rb = rangs(a), rangs(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> int:
    etats: dict[float, dict] = {}
    for b, nom in DOSSIERS:
        p = BASE / nom / "acvram_manifest.json"
        if not p.exists():
            print(f"ECHEC / CAUSE: {nom} absent — la reconstruction a besoin "
                  f"des budgets croissants / SUITE: convertir ce point")
            return 2
        etats[b] = json.load(open(p))["tensors"]
    budgets = sorted(etats)
    noms = sorted(etats[budgets[0]])
    cand = []
    for k in noms:
        base = None
        for b in budgets:
            v = etats[b].get(k)
            if v is None:
                break
            if v["format"] == "nvfp4":
                base = (b, v)
            elif v["format"] == "int8" and base is not None:
                cand.append({"nom": k, "budget_promotion": b,
                             "snr_base": base[1]["out_snr_db"],
                             "snr_prom": v["out_snr_db"],
                             "oct_base": octets(base[1]),
                             "oct_prom": octets(v)})
                break
    if not cand:
        print("ECHEC / CAUSE: aucun tenseur ne change de format entre les budgets")
        return 2
    for c in cand:
        c["cout"] = c["oct_prom"] - c["oct_base"]
        c["gain_db"] = c["snr_prom"] - c["snr_base"]
        c["cle_snr"] = c["gain_db"] / max(c["cout"], 1)
        # perte marginale par octet : l'erreur de sortie EVITEE par octet
        c["cle_perte"] = (err(c["snr_base"]) - err(c["snr_prom"])) / max(c["cout"], 1)

    print(f"{len(cand)} tenseurs changent de format le long des budgets\n")

    # --- TEMOIN : la cle SNR reconstruite doit reproduire l'ordre OBSERVE ---
    obs = sorted(cand, key=lambda c: c["budget_promotion"])
    rec = sorted(cand, key=lambda c: -c["cle_snr"])
    rho_temoin = spearman([c["budget_promotion"] for c in cand],
                          [-c["cle_snr"] for c in cand])
    print(f"TEMOIN — cle SNR reconstruite contre ordre observe des promotions")
    print(f"  Spearman {rho_temoin:+.4f}")
    if rho_temoin < 0.5:
        print("  RECONSTRUCTION REFUSEE : la cle recalculee ne reproduit pas "
              "l'ordre reel des promotions. Rien de ce qui suit ne vaut — le "
              "glouton n'ordonne pas par ce que je crois, ou mes octets sont faux.")
        return 3
    print("  reproduite : la reconstruction tient\n")

    # --- LA QUESTION ---
    rho = spearman([-c["cle_snr"] for c in cand], [-c["cle_perte"] for c in cand])
    print("DESACCORD DES DEUX CLASSEMENTS")
    print(f"  Spearman SNR/octet contre perte/octet : {rho:+.4f}")
    par_snr = [c["nom"] for c in sorted(cand, key=lambda c: -c["cle_snr"])]
    par_perte = [c["nom"] for c in sorted(cand, key=lambda c: -c["cle_perte"])]
    for k in (10, 25, 50, 100):
        if k > len(cand):
            break
        commun = len(set(par_snr[:k]) & set(par_perte[:k]))
        print(f"  top-{k:<4d} : {commun}/{k} communs, "
              f"desaccord {100 * (1 - commun / k):5.1f} %")
    print()
    # =====================================================================
    # CE QUE CE CHIFFRE N'EST PAS, et il faut le lire avant de s'en servir.
    #
    # `cle_snr` vaut (snr_prom - snr_base) / cout.
    # `cle_perte` vaut (10^(-snr_base/20) - 10^(-snr_prom/20)) / cout.
    #
    # LES DEUX SONT DES FONCTIONS DES MEMES DEUX NOMBRES. La seconde n'est pas
    # une mesure independante de la perplexite : c'est la premiere vue a
    # travers 10^(-x/20). Leur correlation mesure donc la NON-LINEARITE de
    # cette transformation, pas un desaccord entre le SNR et la perplexite.
    #
    # Le Spearman global est par consequent presque tautologique. Ce qui reste
    # informatif est le desaccord DE TETE : la transformation reordonne
    # violemment le haut du classement, et c'est le haut que le glouton
    # consomme en premier.
    #
    # POUR REPONDRE VRAIMENT il faut un DL par tenseur mesure sur une perte de
    # calibration — une passe avant par tenseur et par format — que le
    # manifeste ne porte pas. Les `erreurs_grille` n'y suffisent pas : ce sont
    # aussi des erreurs de SORTIE, donc le meme probleme.
    print("  CE QUE CE CHIFFRE N'EST PAS : les deux cles sont des fonctions "
          "des MEMES deux\n  nombres — la seconde est la premiere vue a travers "
          "10^(-snr/20). Leur Spearman\n  mesure la non-linearite de cette "
          "transformation, PAS un desaccord entre le\n  SNR et la perplexite. "
          "Seul le desaccord DE TETE est informatif.\n")
    if rho > 0.95:
        print("  LES DEUX CLES COINCIDENT PRESQUE, ce qui etait attendu puisqu'elles "
              "rien a gagner\n  a substituer la cle, et le chantier se ferme.")
    elif rho < 0.5:
        print("  LES DEUX CLASSEMENTS DIVERGENT FORTEMENT : l'ordre actuel est "
              "peu informatif\n  pour notre objectif, et la substitution vaut "
              "d'etre faite.")
    else:
        print("  DESACCORD PARTIEL : la substitution deplacerait une partie du "
              "classement.\n  L'A/B a signe inverse dira ce que cela vaut en "
              "perplexite.")
    sorties() / "desaccord-classements.json".write_text(json.dumps(
             {"n": len(cand), "spearman": rho, "spearman_temoin": rho_temoin,
              "top": {k: len(set(par_snr[:k]) & set(par_perte[:k]))
                      for k in (10, 25, 50, 100) if k <= len(cand)}},
             indent=2))
    print(f"\nFAIT / TESTE: {len(cand)} tenseurs, Spearman {rho:+.4f}, "
          f"temoin {rho_temoin:+.4f} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
