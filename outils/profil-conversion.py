"""Pièce 23 — où passe le temps d une conversion NVFP4, et la recherche
d échelle domine-t-elle assez pour qu on la vectorise ?

**Ce que les journaux existants ne donnent pas** : le journal par tenseur
(`convert._journal_tenseurs`, `convert.py:680-699`) n est actif que sur
processeur ou sous `ACVRAM_JOURNAL_TENSEURS=1` (`convert.py:686-690`) ; aucune
conversion de carte du 21-22/09 ne l a activé. Le seul journal complet
retrouvé (30B-VL, `qvl30b-reconv-identite-22-09.log`) porte une durée totale
— 475,4 s — et des lignes de progression SANS horodatage. Le profil par phase
ne peut donc pas être reconstruit a posteriori : il se MESURE, et un chiffre
reconstruit pour combler ce trou serait précisément ce qu on ne publie pas.

Ce que cet outil mesure à la place, à sec et sans carte : le coût de chaque
phase PAR TENSEUR, sur des poids synthétiques aux formes réelles du modèle
visé (lues dans un manifeste existant, `--manifeste`, ou données par
`--formes`), puis l extrapolation au modèle entier par comptage.
Phases chronométrées, dans l ordre de `convert_checkpoint` :
  * `recherche`   — `search_channel_scales` (grille AWQ de n_grid+1 points,
                    chacun un quant/déquant complet du tenseur) ;
  * `quant_max6`  — `quantize_nvfp4(echelle="max6")`, le chemin par défaut ;
  * `quant_4sur6` — `quantize_nvfp4(echelle="4sur6")` : DEUX candidats par
                    bloc plus la comparaison de MSE ; la différence des deux
                    est le prix exact de Four Over Six ;
  * `serialisation` — `safetensors.torch.save` des sorties (proxy d écriture).
La lecture disque n est pas simulée : elle dépend du support, et le journal du
30B-VL ne permet pas de l isoler — dit ici plutôt que deviné.

Décision que l outil sert (écrite avant la mesure) : la vectorisation de la
recherche ne vaut d être tentée que si `recherche` ≥ 50 % du temps par
tenseur ; entre 25 et 50 %, gain plafonné, à peser ; sous 25 %, la réponse est
non et la pièce 15 (scission de convert.py) passe devant.

Usage : python outils/profil-conversion.py [--manifeste ALIAS] [--rep 3] [--json S]
        (processeur seulement, ≤ 2 min ; --formes N,K[:compte] répétable)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import torch  # noqa: E402

SEUIL_VECTORISER = 0.50
SEUIL_PESER = 0.25


def formes_du_manifeste(chemin: str) -> dict:
    """{(N, K) : nombre de tenseurs} des poids 2-D quantifiés d un alias."""
    m = json.load(open(os.path.join(chemin, "acvram_manifest.json")))
    compte: dict = {}
    for nom, t in m.get("tensors", {}).items():
        forme = t.get("shape")
        if not forme or len(forme) != 2 or t.get("format") in ("bf16", "fp16", None):
            continue
        cle = (int(forme[0]), int(forme[1]))
        compte[cle] = compte.get(cle, 0) + 1
    return compte


def chrono(f, rep: int) -> float:
    """Médiane de `rep` appels, en secondes."""
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def mesurer_forme(n: int, k: int, rep: int, n_grid: int = 20, lignes_max: int = 256) -> dict:
    """Mesure sur AU PLUS `lignes_max` lignes, puis met à l échelle par n / m.
    Toutes ces phases sont linéaires en lignes (même grille, même arrondi, une
    ligne n influe pas sur une autre) — et le contrôle `linearite` le vérifie
    au lieu de le supposer : m/2 lignes doivent coûter la moitié de m, à 20 %
    près. Hors de cette bande, l extrapolation est déclarée non valable."""
    from acvram.quant.calibrate import ActStats, search_channel_scales
    from acvram.quant.nvfp4 import quantize_nvfp4
    g = torch.Generator().manual_seed(n * 100003 + k)
    m = min(n, lignes_max) if lignes_max else n      # 0 = tenseur entier
    w = (torch.randn(m, k, generator=g) * 0.05).to(torch.bfloat16)
    st = ActStats(mean_abs=torch.rand(k, generator=g) + 0.02, max_abs=None, n_samples=1024)
    echelle = n / m
    r = {"forme": [n, k], "octets_bf16": n * k * 2, "lignes_mesurees": m, "facteur_echelle": round(echelle, 3)}
    wf = w.float()
    r["recherche"] = echelle * chrono(lambda: search_channel_scales(wf, st, "nvfp4", group_size=16,
                                                                    n_grid=n_grid, quantize_activation_nvfp4=True), rep)
    r["quant_max6"] = echelle * chrono(lambda: quantize_nvfp4(wf, echelle="max6"), rep)
    r["quant_4sur6"] = echelle * chrono(lambda: quantize_nvfp4(wf, echelle="4sur6"), rep)
    r["surcout_4sur6"] = r["quant_4sur6"] - r["quant_max6"]
    q = quantize_nvfp4(wf, echelle="max6")
    from safetensors.torch import save
    r["serialisation"] = echelle * chrono(lambda: save({"q": q.qweight, "s": q.block_scale.view(torch.uint8)}), rep)
    if echelle > 1 and m >= 32:                  # contrôle de linéarité : seulement si l on extrapole
        demi = chrono(lambda: search_channel_scales(wf[: m // 2], st, "nvfp4", group_size=16,
                                                    n_grid=n_grid, quantize_activation_nvfp4=True), 1)
        rapport = (r["recherche"] / echelle) / max(demi, 1e-9)
        r["linearite"] = round(rapport, 2)
        r["extrapolation_valable"] = 1.6 <= rapport <= 2.4
    r["total_max6"] = r["recherche"] + r["quant_max6"] + r["serialisation"]
    r["part_recherche"] = r["recherche"] / r["total_max6"]
    return r


def verdict(part: float) -> str:
    if part >= SEUIL_VECTORISER:
        return (f"VECTORISER : la recherche pèse {part:.0%} ≥ {SEUIL_VECTORISER:.0%} du temps par tenseur — "
                "une grille vectorisée (les n_grid+1 candidats en un lot) vaut d être écrite, au bit près "
                "(même arrondi, même ordre de comparaison), avec son test d équivalence dans le même commit")
    if part >= SEUIL_PESER:
        return (f"À PESER : recherche {part:.0%}, entre {SEUIL_PESER:.0%} et {SEUIL_VECTORISER:.0%} — le gain "
                "maximal est borné par cette part ; la pièce 15 (scission de convert.py) rend plus sûrement")
    return (f"NON : la recherche ne pèse que {part:.0%} < {SEUIL_PESER:.0%} — vectoriser ne peut pas rendre ce "
            "qu elle ne coûte pas ; passer à la pièce 15 (scission de convert.py par phases)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifeste", help="alias converti dont on reprend les formes et leur nombre")
    ap.add_argument("--formes", action="append", default=[], metavar="N,K[:C]",
                    help="forme à mesurer, avec son nombre d occurrences (répétable)")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--elements-max", type=int, default=20_000_000, metavar="N",
                    help="au-delà de N éléments, la forme n est pas mesurée (et pas extrapolée) : "
                         "elle est nommée dans `formes_non_mesurees`")
    ap.add_argument("--lignes-max", type=int, default=0,
                    help="lignes réellement mesurées par forme (le reste est extrapolé linéairement, "
                         "avec son contrôle) ; 0 = mesurer le tenseur entier")
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    compte: dict = {}
    if a.manifeste:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from racine_modeles import racine_modeles
        d = a.manifeste if os.path.isdir(a.manifeste) else os.path.join(racine_modeles(), a.manifeste)
        compte = formes_du_manifeste(d)
    for f in a.formes:
        forme, _, c = f.partition(":")
        n, k = (int(x) for x in forme.split(","))
        compte[(n, k)] = compte.get((n, k), 0) + (int(c) if c else 1)
    if not compte:
        sys.exit("ÉCHEC / CAUSE : ni --manifeste ni --formes / SUITE : donner l un des deux")
    # les cinq formes les plus nombreuses portent l essentiel du temps ; celles
    # qui dépassent `--elements-max` ne sont PAS extrapolées (le contrôle de
    # linéarité du 22/09 a montré que la recherche ne l est pas sur processeur :
    # la moitié des lignes coûte 0,8 fois le tout, pas 0,5) — elles sont
    # nommées comme non mesurées plutôt que devinées.
    grandes = sorted(compte.items(), key=lambda kv: -kv[1] * kv[0][0] * kv[0][1])[:5]
    trop_grandes = [(f, c) for f, c in grandes if f[0] * f[1] > a.elements_max]
    grandes = [(f, c) for f, c in grandes if f[0] * f[1] <= a.elements_max]
    mesures = []
    tot = {"recherche": 0.0, "quant_max6": 0.0, "quant_4sur6": 0.0, "serialisation": 0.0, "tenseurs": 0}
    for (n, k), c in grandes:
        m = mesurer_forme(n, k, a.rep, a.n_grid, a.lignes_max)
        m["occurrences"] = c
        mesures.append(m)
        for ph in ("recherche", "quant_max6", "quant_4sur6", "serialisation"):
            tot[ph] += m[ph] * c
        tot["tenseurs"] += c
    total_max6 = tot["recherche"] + tot["quant_max6"] + tot["serialisation"]
    part = tot["recherche"] / total_max6 if total_max6 else 0.0
    r = {"formes_mesurees": mesures, "tenseurs_couverts": tot["tenseurs"],
         "secondes_modele_max6": round(total_max6, 1),
         "secondes_modele_4sur6": round(total_max6 - tot["quant_max6"] + tot["quant_4sur6"], 1),
         "part_recherche": round(part, 4),
         "part_quantification": round(tot["quant_max6"] / total_max6, 4) if total_max6 else None,
         "part_serialisation": round(tot["serialisation"] / total_max6, 4) if total_max6 else None,
         "surcout_4sur6_pct": round(100 * (tot["quant_4sur6"] - tot["quant_max6"]) / total_max6, 2) if total_max6 else None,
         "seuils": {"vectoriser": SEUIL_VECTORISER, "peser": SEUIL_PESER},
         "verdict": verdict(part),
         "extrapolation_valable": all(m.get("extrapolation_valable", True) for m in mesures),
         "formes_non_mesurees": [{"forme": list(f), "occurrences": c, "elements": f[0] * f[1]}
                                 for f, c in trop_grandes]}
    if not r["extrapolation_valable"]:
        r["verdict"] = ("NON JUGÉ : le contrôle de linéarité a échoué sur au moins une forme "
                        "(la moitié des lignes ne coûte pas la moitié) — refaire avec --lignes-max 0")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    print(f"[profil] {len(mesures)} formes, {tot['tenseurs']} tenseurs couverts · processeur, {a.rep} rép")
    for m in mesures:
        print(f"    {str(m['forme']):>14s} ×{m['occurrences']:<6d} recherche {m['recherche'] * 1e3:7.1f} ms · "
              f"quant max6 {m['quant_max6'] * 1e3:6.1f} · 4sur6 {m['quant_4sur6'] * 1e3:6.1f} "
              f"(+{m['surcout_4sur6'] * 1e3:.1f}) · sérialisation {m['serialisation'] * 1e3:6.1f} · "
              f"recherche {m['part_recherche']:.0%}"
              + (f" · {m['lignes_mesurees']}/{m['forme'][0]} lignes ×{m['facteur_echelle']}"
                 f" linéarité {m.get('linearite')}" if m["facteur_echelle"] > 1 else ""))
    print(f"  modèle (formes couvertes) : {r['secondes_modele_max6']} s en max6, {r['secondes_modele_4sur6']} s en 4sur6 "
          f"(+{r['surcout_4sur6_pct']} %)")
    if r["formes_non_mesurees"]:
        print("  non mesurées (au-delà de --elements-max, jamais extrapolées) : "
              + ", ".join(f"{m['forme']}×{m['occurrences']}" for m in r["formes_non_mesurees"]))
    print(f"  parts : recherche {r['part_recherche']:.0%} · quantification {r['part_quantification']:.0%} · "
          f"sérialisation {r['part_serialisation']:.0%}")
    print(f"  verdict : {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
