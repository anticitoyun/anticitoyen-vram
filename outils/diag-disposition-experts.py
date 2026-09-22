"""Pourquoi un converti MoE charge-t-il en `experts_layout=naturel` au lieu de
`marlin` ? — à sec, depuis le manifeste et les échelles d activation, sans
carte et sans charger le modèle.

La disposition Marlin (préfill groupé + GEMV (b) du décodage) est refusée par
`engine/moe.py:446` quand `up_distinct` est vrai, c est-à-dire quand les tables
AWQ de `gate_proj` et `up_proj` ne sont pas le MÊME objet après la fusion de
`moe.py:336-337` :

    if g is not None and u is not None and torch.equal(g, u):
        awq["up_proj"] = g                       # fusion
    awq["up_distinct"] = not (awq.get("up_proj") is g)

Un SEUL expert dont l échelle de gate diffère de celle d up suffit : la table
est [E, K], `torch.equal` est global. Et `awq[nom]` vaut None quand la table
entière vaut 1 (`moe.py:257-258`) — un converti SANS échelle d experts, ou
dont toutes les échelles sont l identité, a donc `up_distinct` faux et garde
Marlin. C est la différence entre deux convertis qui se ressemblent.

Cet outil dit, pour un alias : `has_act_scale`, la part d experts dont
gate ≠ up par couche, et la disposition PRÉDITE avec sa raison. Il ne remplace
pas le chargement : il dit ce que le chargement fera, et pourquoi.

Usage : python outils/diag-disposition-experts.py ALIAS [ALIAS2 …] [--couches 0,1,24,47] [--json S]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open


def analyser(dossier: str, couches: list[int]) -> dict:
    m = json.load(open(os.path.join(dossier, "acvram_manifest.json")))
    wm, tenseurs = m["weight_map"], m.get("tensors", {})
    cle0 = next((k for k in tenseurs if ".experts.0.gate_proj" in k), None)
    if cle0 is None:
        return {"alias": os.path.basename(dossier), "moe": False,
                "verdict": "pas un MoE (aucun expert dans le manifeste)"}
    has = bool(tenseurs[cle0].get("has_act_scale"))
    ouverts: dict = {}

    def lire(cle):
        f = wm.get(cle)
        if f is None:
            return None
        if f not in ouverts:
            ouverts[f] = safe_open(os.path.join(dossier, f), framework="pt", device="cpu")
        return ouverts[f].get_tensor(cle)

    E = 1 + max((int(k.split(".experts.")[1].split(".")[0]) for k in tenseurs if ".experts." in k), default=0)
    par_couche = {}
    for c in couches:
        diff = manquants = unite = 0
        ecarts = []
        for e in range(E):
            g = lire(f"model.layers.{c}.mlp.experts.{e}.gate_proj.weight.act_scale")
            u = lire(f"model.layers.{c}.mlp.experts.{e}.up_proj.weight.act_scale")
            if g is None or u is None:
                manquants += 1
                continue
            if bool(torch.all(g == 1)) and bool(torch.all(u == 1)):
                unite += 1
            if not torch.equal(g, u):
                diff += 1
                ecarts.append(float((g.float() - u.float()).norm() / g.float().norm().clamp(min=1e-30)))
        ecarts.sort()
        par_couche[c] = {"experts": E, "gate_ne_up": diff, "sans_echelle": manquants, "echelle_unite": unite,
                         "ecart_median": round(ecarts[len(ecarts) // 2], 4) if ecarts else 0.0,
                         "ecart_max": round(ecarts[-1], 4) if ecarts else 0.0}
    distinct = any(v["gate_ne_up"] for v in par_couche.values())
    toutes_unite = all(v["sans_echelle"] + v["echelle_unite"] == v["experts"] for v in par_couche.values())
    if not has or toutes_unite:
        verdict = ("marlin PRÉDIT : aucune table AWQ effective (has_act_scale="
                   f"{has}, toutes à l identité) — `awq[nom]` vaut None, up_distinct faux (moe.py:257, 337)")
    elif distinct:
        pire = max(par_couche.values(), key=lambda v: v["gate_ne_up"])
        verdict = ("naturel PRÉDIT : tables AWQ gate/up distinctes — jusqu à "
                   f"{pire['gate_ne_up']}/{pire['experts']} experts par couche, écart relatif médian "
                   f"{pire['ecart_median']} ; `torch.equal` est global sur [E, K], un seul expert suffit "
                   "(moe.py:336-337) et `_construire_marlin` refuse (moe.py:446, raison « gate/up à "
                   "entrées distinctes »). Cause en amont : les experts MoE sont EXCLUS de la pré-passe "
                   "d alpha commun gate/up (quant/convert.py:543-544, « hors experts MoE »)")
    else:
        verdict = "marlin PRÉDIT : tables AWQ présentes mais gate == up partout (fusion moe.py:336)"
    return {"alias": os.path.basename(dossier), "moe": True, "has_act_scale": has,
            "experts_repli": m.get("experts_repli"), "experts_sans_stats": m.get("experts_sans_stats"),
            "par_couche": par_couche, "up_distinct_predit": bool(has and distinct and not toutes_unite),
            "verdict": verdict}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("alias", nargs="+")
    ap.add_argument("--couches", default="0,1,24,47")
    ap.add_argument("--json")
    a = ap.parse_args()
    couches = [int(x) for x in a.couches.split(",")]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from racine_modeles import racine_modeles
    out = []
    for al in a.alias:
        d = al if os.path.isdir(al) else os.path.join(racine_modeles(), al)
        r = analyser(d, couches)
        out.append(r)
        print(f"[disposition] {r['alias']}")
        if not r["moe"]:
            print(f"    {r['verdict']}"); continue
        print(f"    has_act_scale={r['has_act_scale']} · experts_repli={r['experts_repli']} · "
              f"sans_stats={r['experts_sans_stats']}")
        for c, v in r["par_couche"].items():
            print(f"    couche {c:3d} : gate≠up {v['gate_ne_up']:4d}/{v['experts']} · sans échelle "
                  f"{v['sans_echelle']:4d} · identité {v['echelle_unite']:4d} · écart médian {v['ecart_median']:.4f} "
                  f"max {v['ecart_max']:.4f}")
        print(f"    {r['verdict']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
