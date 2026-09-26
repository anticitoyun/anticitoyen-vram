#!/usr/bin/env python3
"""`absolu` exact, sans conversion de ma part : un seul dossier portant
`out_ref_norm` suffit pour les 225 candidats.

POURQUOI UN SEUL SUFFIT, et c'est une lecture de `calibrate.py` :

    w     = weight.detach().to(float32)      le poids SOURCE
    probe = stats.mean_abs                   les statistiques d'activation
    y_ref = diag(probe) @ w.t()
    out_ref_norm = ||y_ref||

**`out_ref_norm` ne depend PAS du format cible.** Il se calcule sur le poids
source et la sonde de calibration ; `deq` n'intervient que dans le NUMERATEUR
de l'erreur. Donc l'echelle d'un tenseur est la meme qu'il finisse en nvfp4 ou
en int8, et un dossier quelconque produit avec le code du 10/09 la donne pour
chacun de ses tenseurs.

Consequence : je n'ai pas besoin de la carte. La conversion `erreur` de
poste4 porte le champ pour les 225 candidats, et ma simulation devient exacte
au lieu d'etre un substitut par ||w||_F.

RESERVE, verifiee par le script : si `hadamard_block` n'est pas nul, `w` est
tourne avant le calcul et l'echelle n'est plus comparable d'un dossier a
l'autre. Le script REFUSE dans ce cas au lieu de melanger deux reperes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402

BASE = Path(MODELES)
GIO = 1024 ** 3
BUDGET = 6.00
A_PROMUS, A_PPL = 198, 5.4918
B_PROMUS, B_PPL = 149, 5.4482
PLAFOND_PPL = 5.4144
# dossiers a budgets croissants, pour reconstruire les DEUX SNR de chaque tenseur
DOS = [(0.0, "Llama-2-7b-nvfp4"), (4.50, "Llama-2-7b-quota-4g50"),
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


err = lambda s: 10.0 ** (-s / 20.0)


def spearman(x, y):
    n = len(x)
    def rg(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for i, j in enumerate(o):
            r[j] = i + 1
        return r
    a, b = rg(x), rg(y)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    da = sum((p - ma) ** 2 for p in a) ** 0.5
    db = sum((q - mb) ** 2 for q in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: simuler-absolu-exact.py <dossier portant out_ref_norm>")
        return 2
    src = BASE / sys.argv[1]
    p = src / "acvram_manifest.json"
    if not p.exists():
        print(f"ECHEC / CAUSE: {p} absent")
        return 2
    t_ech = json.loads(p.read_text())["tensors"]
    ech = {k: v["out_ref_norm"] for k, v in t_ech.items()
           if "out_ref_norm" in v}
    if not ech:
        print(f"ECHEC / CAUSE: {sys.argv[1]} ne porte aucun out_ref_norm — "
              f"dossier produit avant l'ajout du champ / SUITE: prendre un "
              f"dossier converti avec le code du 10/09 ou plus recent")
        return 3
    had = {v.get("hadamard_block", 0) for v in t_ech.values()}
    if had - {0}:
        print(f"REFUS : hadamard_block non nul {had - {0}} — `w` est tourne "
              f"avant le calcul de l'echelle, qui n'est alors plus comparable "
              f"d'un dossier a l'autre. Melanger deux reperes donnerait un "
              f"classement sans signification.")
        return 3
    print(f"echelles lues dans {sys.argv[1]} : {len(ech)} tenseurs, "
          f"hadamard_block nul partout")
    v = list(ech.values())
    v.sort()
    q = [v[int(len(v) * f)] for f in (0.05, 0.25, 0.5, 0.75, 0.95)]
    print(f"  ||y_ref||  min {v[0]:.4g}  max {v[-1]:.4g}  "
          f"rapport {v[-1] / max(v[0], 1e-12):.1f}x")
    print(f"  quantiles 5/25/50/75/95 : " + "  ".join(f"{x:.4g}" for x in q))
    print(f"  rapport entre quartiles 25 et 95 : {q[4] / max(q[1], 1e-12):.2f}x"
          f"   <- a comparer au 2,1x du substitut ||w||_F")

    etats = {}
    for b, nom in DOS:
        pp = BASE / nom / "acvram_manifest.json"
        if not pp.exists():
            print(f"ECHEC / CAUSE: {nom} absent")
            return 2
        etats[b] = json.loads(pp.read_text())
    bs = sorted(etats)
    tens = {b: etats[b]["tensors"] for b in bs}
    plancher = etats[6.00]["budget"]["plancher_gib"]
    cand = []
    for k in sorted(tens[bs[0]]):
        base = None
        for b in bs:
            vv = tens[b].get(k)
            if vv is None:
                break
            if vv["format"] == "nvfp4":
                base = vv
            elif vv["format"] == "int8" and base and k in ech:
                cand.append({"nom": k, "sb": base["out_snr_db"],
                             "sp": vv["out_snr_db"],
                             "cout": max(octets(vv) - octets(base), 1),
                             "ech": ech[k]})
                break
    print(f"\n{len(cand)} candidats avec echelle exacte")
    reste_total = BUDGET * GIO - plancher * GIO
    cles = {
        "snr": lambda c: -((c["sp"] - c["sb"]) / c["cout"]),
        "inverse": lambda c: +((c["sp"] - c["sb"]) / c["cout"]),
        "erreur": lambda c: -((err(c["sb"]) - err(c["sp"])) / c["cout"]),
        "absolu": lambda c: -(c["ech"] * (err(c["sb"]) - err(c["sp"])) / c["cout"]),
        "base_croissant": lambda c: c["sb"],
    }
    res = {}
    for nom, cle in cles.items():
        reste, pr = reste_total, []
        for c in sorted(cand, key=cle):
            if c["cout"] <= reste:
                pr.append(c["nom"])
                reste -= c["cout"]
        res[nom] = pr
        print(f"  {nom:16s} {len(pr):3d} promus, {reste / 2**20:7.1f} Mio inemployes")
    ecart = abs(len(res["snr"]) - A_PROMUS)
    print(f"\nTEMOIN — cle snr simulee contre les {A_PROMUS} du bras A converti : "
          f"{len(res['snr'])}, ecart {ecart}")
    if ecart > 6:
        print("  SIMULATION REFUSEE : ecart au-dela de la discretisation.")
        return 3
    r_e = [c["nom"] for c in sorted(cand, key=cles["erreur"])]
    r_a = [c["nom"] for c in sorted(cand, key=cles["absolu"])]
    print(f"\n  absolu contre erreur : top-10 "
          f"{len(set(r_a[:10]) & set(r_e[:10]))}/10, top-100 "
          f"{len(set(r_a[:100]) & set(r_e[:100]))}/100, ensembles "
          f"{len(set(res['absolu']) & set(res['erreur']))}/"
          f"{max(len(res['absolu']), len(res['erreur']))}")
    pe = {n: i for i, n in enumerate(r_e)}
    pa = {n: i for i, n in enumerate(r_a)}
    print(f"  correlation echelle EXACTE / deplacement de rang : "
          f"{spearman([c['ech'] for c in cand], [pe[c['nom']] - pa[c['nom']] for c in cand]):+.4f}")
    print(f"\n  A COMPARER AU SUBSTITUT ||w||_F : 195 promus, top-100 79/100, "
          f"correlation +0,4653.\n  L'ecart entre les deux dit ce que mon "
          f"approximation coutait.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
