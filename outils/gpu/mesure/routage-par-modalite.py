"""Pièce 27(a) : table 48 × 128 du routage MoE par MODALITÉ, depuis une trace
v2 (`ACVRAM_TRACE_ROUTAGE=<fichier>` sur un lot multimodal).

Pour chaque (couche, expert) : nombre de sélections top-k et masse des poids
de porte, par modalité (i = image, t = texte, s = spécial), et le **ratio
conditionnel** Q^m = (part du trafic de la modalité m qui va à cet expert) ÷
(part attendue si le routage ignorait la modalité, = part de cet expert tous
trafics confondus). Q^i ≫ 1 : expert spécialisé image ; Q^i = 0 : jamais vu
sur une position image.

Sortie : `<sortie>.json` (table complète, totaux, experts par classe) et
`<sortie>.tsv` (couche, expert, n_i, n_t, n_s, masse_i, masse_t, masse_s,
Q_i, Q_t). Avec `--froids <manifeste>` (manifeste d un converti portant
`experts_sans_stats_liste`, pièce 25) : la part des experts froids qui
n apparaissent QUE sur des positions image — **prédiction écrite avant :
≥ 80 % → la voie texte est fermée pour eux** (le corpus texte ne les
atteindra jamais, il faut calibrer sur des images) ; < 50 % → la voie texte
reste ouverte et le corpus de calibration est en cause.

Usage : python outils/gpu/mesure/routage-par-modalite.py trace.txt sortie [--experts 128] [--froids manifest.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("ACVRAM_ARBRE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))

SEUIL_FERME = 0.80
SEUIL_OUVERT = 0.50


def agreger(trace: str) -> dict:
    from acvram.memory.trace_routage import relire_modalites
    n = defaultdict(lambda: defaultdict(int))           # (couche, expert) → modalité → sélections
    masse = defaultdict(lambda: defaultdict(float))
    tot_m = defaultdict(int)
    couches, experts = set(), set()
    for _jeton, couche, m, paires in relire_modalites(trace):
        couches.add(couche)
        for e, w in paires:
            experts.add(e)
            n[(couche, e)][m] += 1
            masse[(couche, e)][m] += float(w)
            tot_m[m] += 1
    return {"n": n, "masse": masse, "tot_m": dict(tot_m), "couches": sorted(couches), "experts": sorted(experts)}


def table(agg: dict, nb_experts: int) -> dict:
    n, masse, tot_m = agg["n"], agg["masse"], agg["tot_m"]
    total = sum(tot_m.values()) or 1
    lignes = {}
    for (couche, e), par_m in sorted(n.items()):
        tot_e = sum(par_m.values())
        part_e = tot_e / total                            # part de cet expert, tous trafics
        q = {}
        for m, k in tot_m.items():
            if k:
                q[m] = round((par_m.get(m, 0) / k) / part_e, 4) if part_e else 0.0
        lignes[f"{couche}/{e}"] = {"couche": couche, "expert": e, "n": dict(par_m),
                                   "masse": {m: round(v, 4) for m, v in masse[(couche, e)].items()},
                                   "Q": q, "total": tot_e}
    return {"lignes": lignes, "total_selections": total, "par_modalite": tot_m,
            "couches": len(agg["couches"]), "experts_vus": len(agg["experts"]), "experts_declares": nb_experts}


def experts_froids(manifeste: str) -> set:
    m = json.load(open(manifeste))
    out = set()
    for couche, projs in (m.get("experts_sans_stats_liste") or {}).items():
        for _proj, es in projs.items():
            for e in es:
                out.add((int(couche), int(e)))
    return out


def verdict_froids(t: dict, froids: set) -> dict:
    """Part des experts froids qui n apparaissent QUE sur des positions image."""
    vus = {(l["couche"], l["expert"]): l for l in t["lignes"].values()}
    image_seul, vus_froids, jamais = 0, 0, 0
    for cle in froids:
        l = vus.get(cle)
        if l is None:
            jamais += 1
            continue
        vus_froids += 1
        if l["n"].get("i", 0) > 0 and l["n"].get("t", 0) == 0:
            image_seul += 1
    part = image_seul / vus_froids if vus_froids else 0.0
    if not vus_froids:
        v = f"NON JUGÉ : aucun des {len(froids)} experts froids n apparaît dans cette trace (élargir le lot)"
    elif part >= SEUIL_FERME:
        v = (f"TENU : {part:.0%} des experts froids vus ne sont routés que sur des positions image "
             f"(≥ {SEUIL_FERME:.0%}) — la voie texte est fermée pour eux, il faut calibrer sur des images")
    elif part < SEUIL_OUVERT:
        v = (f"RÉFUTÉ : {part:.0%} seulement ne sont vus que sur image (< {SEUIL_OUVERT:.0%}) — "
             f"la voie texte reste ouverte, le corpus de calibration est en cause")
    else:
        v = f"MARGINAL : {part:.0%} entre {SEUIL_OUVERT:.0%} et {SEUIL_FERME:.0%} — les deux voies comptent"
    return {"froids_declares": len(froids), "froids_vus": vus_froids, "froids_jamais_vus": jamais,
            "froids_image_seule": image_seul, "part_image_seule": round(part, 4), "verdict": v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("sortie", help="préfixe : <sortie>.json et <sortie>.tsv")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--froids", help="manifeste d un converti (experts_sans_stats_liste)")
    a = ap.parse_args()
    t = table(agreger(a.trace), a.experts)
    if a.froids:
        t.update(verdict_froids(t, experts_froids(a.froids)))
    json.dump(t, open(a.sortie + ".json", "w"), indent=1)
    with open(a.sortie + ".tsv", "w") as f:
        f.write("couche\texpert\tn_i\tn_t\tn_s\tmasse_i\tmasse_t\tmasse_s\tQ_i\tQ_t\n")
        for l in t["lignes"].values():
            f.write(f"{l['couche']}\t{l['expert']}\t{l['n'].get('i', 0)}\t{l['n'].get('t', 0)}\t{l['n'].get('s', 0)}\t"
                    f"{l['masse'].get('i', 0.0)}\t{l['masse'].get('t', 0.0)}\t{l['masse'].get('s', 0.0)}\t"
                    f"{l['Q'].get('i', 0.0)}\t{l['Q'].get('t', 0.0)}\n")
    print(f"[routage] {t['couches']} couches × {t['experts_vus']}/{t['experts_declares']} experts vus ; "
          f"sélections {t['total_selections']} ({t['par_modalite']}) → {a.sortie}.json/.tsv")
    if "verdict" in t:
        print(f"  froids : {t['froids_vus']}/{t['froids_declares']} vus, {t['froids_image_seule']} image seule "
              f"({t['part_image_seule']:.0%}), jamais vus {t['froids_jamais_vus']}")
        print(f"  verdict : {t['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
