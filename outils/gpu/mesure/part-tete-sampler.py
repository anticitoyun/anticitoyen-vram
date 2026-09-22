"""Pièce 39 : part de la TÊTE et de l ÉCHANTILLONNAGE dans le pas servi,
depuis une trace nsys du service (b=12, défauts 0.6.35 : `sampler=graphe`,
donc l argmax est DANS le graphe, entre la tête et la fin du pas).

Un pas = 48 `_route_fusee_kernel` (le marqueur de `familles-noyaux`). Dans
chaque pas : la TÊTE est le GEMM de sortie (`cutlass…wmma` chez nous, 1 par
pas, [12, 151 936] depuis l état caché) ; l ÉCHANTILLONNAGE glouton capturé
est ce qui suit la tête jusqu à la fin du pas (argmax, gather, logsumexp,
écriture du tampon [2, n] : `reduce_kernel`, `vectorized_elementwise`,
`unrolled_elementwise` en queue de pas).

Sortie : µs par pas et % du pas pour la tête, l échantillonnage et le reste,
avec les noyaux nommés un à un ; c est le chiffre que la pièce 39 demande
(« que gagnerait une tête int8 plus rapide ou un argmax fusionné ? »).

Usage : python outils/gpu/mesure/part-tete-sampler.py trace_cuda_gpu_trace.csv [--couches 48] [--json S]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict

MARQUEUR = "_route_fusee_kernel"
# La TÊTE se reconnaît à sa GRILLE, pas à son nom : chez acvram elle passe par
# le noyau étroit int8 (`_etroit_reduit_kernel`, grille [2374×1] = 151 936/64)
# et `cutlass…wmma` est le GEMM du ROUTEUR, une fois par couche (22/09,
# verdict 41 et poste2 63767f45 : le nom seul donnait 2,6 µs de « tête » et
# 94 % de « sampler »). Un noyau qui balaie le vocabulaire a une grille X
# d au moins quelques centaines de tuiles et dure des dizaines de µs.
TETE_GRILLE_MIN = 512
TETE_US_MIN = 20.0
TETE = r"cutlass.*wmma|lm_head|tete_|_etroit_reduit_kernel"
# Ce qui suit la tête dans la fenêtre contient AUSSI le début du pas suivant
# (la fenêtre court d une route à l autre) : l échantillonnage s arrête au
# premier noyau de couche rencontré.
COUCHE = r"_etroit_reduit_kernel|_partiel|rmsnorm|rope_|kv_write|marlin|moe_|_route_fusee|int8_gemv|nvfp4_"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--couches", type=int, default=48)
    ap.add_argument("--json")
    a = ap.parse_args()
    with open(a.trace) as f:
        r = csv.DictReader(f)
        nom_col = "Name" if "Name" in r.fieldnames else "Kernel Name"
        grille = "GrdX" in r.fieldnames
        noyaux = sorted((int(float(x["Start (ns)"])), int(float(x["Duration (ns)"])), x[nom_col],
                         int(float(x["GrdX"])) if grille and x.get("GrdX") else 0) for x in r)
    bornes = [t for t, _, n, _ in noyaux if MARQUEUR in n][::a.couches]
    if len(bornes) < 6:
        sys.exit(f"INVALIDE : {len(bornes)} pas trouvés — trace trop courte")
    fenetres = list(zip(bornes[:-1], bornes[1:]))[2:-1]
    pas, j = [], 0
    for d, fin in fenetres:
        while j < len(noyaux) and noyaux[j][0] < d:
            j += 1
        k, dans = j, []
        while k < len(noyaux) and noyaux[k][0] < fin:
            dans.append(noyaux[k]); k += 1
        # le DERNIER `cutlass…wmma` du pas, pas le premier (22/09, poste2
        # 63767f45 : le routeur de chaque couche utilise le même noyau cutlass,
        # et prendre le premier faisait passer 47 couches pour « après la tête »
        # → « sampler » à 94 % du pas). La tête est le dernier GEMM de sortie
        # avant la fin du pas ; tout ce qui la suit est l échantillonnage.
        # le DERNIER candidat du pas qui balaie le vocabulaire : grille ≥ seuil
        # (ou, sans grille dans la trace, nom + durée) — jamais le premier
        # (22/09, poste2 63767f45 : le `cutlass…wmma` du ROUTEUR, 1 par couche,
        # passait pour la tête → « sampler » à 94 % du pas), et jamais un noyau
        # trop court pour être une tête [b, vocab] (verdict 41 : la nôtre est
        # l étroit int8 de grille [2374×1], 205 µs).
        i_tete = next((i for i in range(len(dans) - 1, -1, -1)
                       if re.search(TETE, dans[i][2])
                       and (dans[i][3] >= TETE_GRILLE_MIN if dans[i][3] else True)
                       and dans[i][1] / 1e3 >= TETE_US_MIN), None)
        if i_tete is None:
            continue
        tete = dans[i_tete][1] / 1e3
        apres = []
        for x in dans[i_tete + 1:]:
            if re.search(COUCHE, x[2]):
                break
            apres.append(x)
        ech = sum(x[1] for x in apres) / 1e3
        noms_ech = defaultdict(float)
        for _, dd, n, _g in apres:
            noms_ech[re.sub(r"\(.*", "", n).replace("void ", "")[:44]] += dd / 1e3
        pas.append({"total": (fin - d) / 1e3, "noyaux": sum(x[1] for x in dans) / 1e3,
                    "tete": tete, "tete_nom": re.sub(r"\(.*", "", dans[i_tete][2]).replace("void ", "")[:44],
                    "tete_grille": dans[i_tete][3], "echantillon": ech, "n_apres": len(apres), "noms": dict(noms_ech)})
    if not pas:
        sys.exit(f"INVALIDE : aucune tête trouvée dans les pas — motif {TETE}, grille ≥ {TETE_GRILLE_MIN}, "
                 f"durée ≥ {TETE_US_MIN} µs ; nommer le noyau de tête si ce moteur en a un autre")
    med = lambda c: statistics.median(p[c] for p in pas)                    # noqa: E731
    noms = defaultdict(list)
    for p in pas:
        for n, v in p["noms"].items():
            noms[n].append(v)
    out = {"trace": a.trace, "pas_juges": len(pas), "us": {c: round(med(c), 2) for c in ("total", "noyaux", "tete", "echantillon")},
           "part_tete": round(med("tete") / med("total"), 4), "part_echantillon": round(med("echantillon") / med("total"), 4),
           "noyaux_apres_tete": {n: round(statistics.median(v), 2) for n, v in sorted(noms.items(), key=lambda kv: -statistics.median(kv[1]))},
           "lancements_apres_tete": int(statistics.median(p["n_apres"] for p in pas)),
           "tete_nom": pas[len(pas) // 2]["tete_nom"], "tete_grille": pas[len(pas) // 2]["tete_grille"]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(f"[part] {out['pas_juges']} pas · pas {out['us']['total']} µs (noyaux {out['us']['noyaux']})")
    print(f"  tête          {out['us']['tete']:7.2f} µs  {out['part_tete']:.2%} du pas  "
          f"({out['tete_nom']}, grille {out['tete_grille']})")
    print(f"  échantillon   {out['us']['echantillon']:7.2f} µs  {out['part_echantillon']:.2%} du pas  ({out['lancements_apres_tete']} lancements après la tête)")
    for n, v in list(out["noyaux_apres_tete"].items())[:6]:
        print(f"      {n:44s} {v:6.2f} µs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
