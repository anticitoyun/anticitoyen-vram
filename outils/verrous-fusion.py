#!/usr/bin/env python3
"""Tous les verrous de fusion, evalues EN UNE PASSE, pour chaque groupe.

Ce script existe parce que la recherche des verrous s'est faite un par un,
dans l'ordre ou ils bloquaient : biais, puis hadamard, puis format, puis
echelle. Une recherche qui progresse ainsi GARANTIT que l'erreur se repete,
parce qu'a chaque etape le verrou suivant est invisible par construction —
trois conclusions fausses en une journee, chacune tiree d'un verrou leve sur
quatre.

Le correctif n'est pas la vigilance, c'est la structure : la conjonction est
ecrite UNE FOIS et tous ses termes sont evalues ensemble. La regle devient une
structure de donnees au lieu d'une discipline, et elle ne peut plus etre
enfreinte.

Elle rend en prime l'assiette de CHAQUE voie, puisque chaque voie est le
sous-ensemble defini par les termes qu'elle leve.
"""
import collections
import json
import os
import sys

import torch
from safetensors import safe_open

A = "/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram"
OCTETS = {"nvfp4": .5625, "int4_awq": .5625, "int8": 1.0625,
          "bf16": 2.0, "fp16": 2.0, "q3n": .40625}
GROUPES = {"qkv": ("self_attn", ["q", "k", "v"]),
           "gate_up": ("mlp", ["gate", "up"])}


def _echelles_egales(dossier, wm, cles):
    """Les act_scale du groupe sont-elles identiques AU BIT PRES ?

    Rend (verdict, presentes). Une echelle absente n'est pas une echelle
    differente : les deux cas se distinguent et ne se traitent pas pareil.
    """
    kk = [c + ".act_scale" for c in cles]
    presentes = [k for k in kk if k in wm]
    if not presentes:
        return True, False                 # aucune echelle : rien a concilier
    if len(presentes) != len(kk):
        return False, True                 # certaines seulement : incomparable
    ref = None
    for k in kk:
        with safe_open(os.path.join(dossier, wm[k]), framework="pt") as f:
            t = f.get_tensor(k)
        if ref is None:
            ref = t
        elif not torch.equal(ref, t):
            return False, True
    return True, True


def verrous(dossier):
    """Un enregistrement par groupe, avec TOUS ses verrous."""
    p = os.path.join(dossier, "acvram_manifest.json")
    m = json.load(open(p))
    ts, wm = m["tensors"], m.get("weight_map", {})
    out = []
    for c in range(400):
        for genre, (sous, noms) in GROUPES.items():
            cles = [f"model.layers.{c}.{sous}.{n}_proj.weight" for n in noms]
            if not all(k in ts for k in cles):
                continue
            fmts = {str(ts[k].get("format")) for k in cles}
            hads = {ts[k].get("hadamard_block") or 0 for k in cles}
            biais = [k.replace(".weight", ".bias") in wm for k in cles]
            ech_ok, ech_presentes = _echelles_egales(dossier, wm, cles)
            out.append({
                "couche": c, "genre": genre,
                "format_ok": len(fmts) == 1,
                "hadamard_ok": len(hads) == 1,
                "biais_ok": len(set(biais)) == 1,      # tous ou aucun
                "echelle_ok": ech_ok,
                "echelles_presentes": ech_presentes,
                "formats": sorted(fmts),
            })
        if not any(f"model.layers.{c}.self_attn.q_proj.weight" in ts
                   for _ in (0,)):
            break
    return out


def main(argv):
    cibles = argv[1:] or sorted(os.listdir(A))
    total = collections.Counter()
    par_voie = collections.Counter()
    for nom in cibles:
        d = os.path.join(A, nom)
        if not os.path.isfile(os.path.join(d, "acvram_manifest.json")):
            continue
        try:
            gs = verrous(d)
        except Exception as e:                          # noqa: BLE001
            print(f"  {nom} : illisible ({str(e)[:40]})", file=sys.stderr)
            continue
        for g in gs:
            total["groupes"] += 1
            libres = (g["format_ok"] and g["hadamard_ok"]
                      and g["biais_ok"] and g["echelle_ok"])
            if libres:
                par_voie["fusionnent deja"] += 1
                continue
            # quelle voie leverait CE groupe, sans rien approximer ?
            if not g["echelle_ok"]:
                par_voie["bloque par ECHELLE (table AWQ)"] += 1
            elif not g["format_ok"]:
                par_voie["bloque par FORMAT seul (voie 2+1)"] += 1
            elif not g["hadamard_ok"]:
                par_voie["bloque par HADAMARD"] += 1
            else:
                par_voie["bloque par BIAIS"] += 1
    print(f"{'etat':38s} {'groupes':>8s} {'part':>7s}")
    t = total["groupes"] or 1
    for k, n in par_voie.most_common():
        print(f"{k:38s} {n:8d} {100*n/t:6.1f} %")
    print(f"{'TOTAL':38s} {t:8d}")
    print("\nLa voie 2+1 ne s'applique qu'aux groupes bloques par le FORMAT SEUL :")
    print("un groupe dont les echelles different reste refuse quel que soit son format.")


if __name__ == "__main__":
    main(sys.argv)
