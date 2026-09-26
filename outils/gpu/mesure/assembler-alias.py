#!/usr/bin/env python3
"""Pièce 107 : assembler un alias à sec à partir de deux alias convertis — les tenseurs dont le nom correspond à un motif
sont pris chez le DONNEUR (entrée de manifeste + octets), tout le reste chez la BASE (fragments liés, pas copiés).
Un tenseur converti ne dépend que de sa source, de son format et de ses options : l'assemblage rend, pour ces
tenseurs, exactement ce qu'une conversion avec ces formats aurait écrit — sans les 30 minutes de carte ; il isole
UN facteur (projections, tête, v/o) là où deux conversions complètes diffèrent par tout à la fois.

    assembler-alias.py BASE DONNEUR SORTIE --prendre REGEX [--prendre REGEX ...]

Le manifeste de sortie porte `assemblage` (base, donneur, motifs, tenseurs pris) ; `weight_map` pointe les tenseurs
pris vers `acvram-assemble.safetensors`. Aucun chemin absolu n'est écrit dans le manifeste.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("donneur")
    ap.add_argument("sortie")
    ap.add_argument("--prendre", action="append", required=True, help="regex sur le nom du tenseur (manifeste)")
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file

    mb = json.load(open(os.path.join(a.base, "acvram_manifest.json")))
    md = json.load(open(os.path.join(a.donneur, "acvram_manifest.json")))
    motifs = [re.compile(p) for p in a.prendre]
    pris = [n for n in mb["tensors"] if any(m.search(n) for m in motifs)]
    if not pris:
        print("aucun tenseur ne correspond", file=sys.stderr)
        return 2
    manquants = [n for n in pris if n not in md["tensors"]]
    if manquants:
        print(f"absents chez le donneur : {manquants[:5]}", file=sys.stderr)
        return 2
    os.makedirs(a.sortie, exist_ok=True)
    # Un alias assemblé peut servir de base : son fragment `acvram-assemble*.safetensors` doit rester lisible, donc le
    # nouveau fragment prend un nom libre (23/09 : S9 sur S8 écrasait le fragment de S8 par le lien, 576 clés perdues).
    existants = {fn for fn in os.listdir(a.base) if fn.startswith("acvram-assemble")}
    n_frag = 0
    while (f"acvram-assemble{n_frag or ''}.safetensors") in existants:
        n_frag += 1
    NOUVEAU = f"acvram-assemble{n_frag or ''}.safetensors"
    tenseurs, wm = {}, dict(mb["weight_map"])
    ouverts: dict[str, object] = {}
    for n in pris:
        for k in mb["tensors"][n]["keys"]:
            wm.pop(k, None)
        for k in md["tensors"][n]["keys"]:
            fn = md["weight_map"][k]
            if fn not in ouverts:
                ouverts[fn] = safe_open(os.path.join(a.donneur, fn), framework="pt", device="cpu")
            tenseurs[k] = ouverts[fn].get_tensor(k).contiguous()
            wm[k] = NOUVEAU
        mb["tensors"][n] = md["tensors"][n]
    save_file(tenseurs, os.path.join(a.sortie, NOUVEAU))
    mb["weight_map"] = wm
    mb["assemblage"] = {"base": os.path.basename(os.path.normpath(a.base)), "donneur": os.path.basename(os.path.normpath(a.donneur)),
                        "motifs": a.prendre, "tenseurs_pris": len(pris), "cles_prises": len(tenseurs), "fragment": NOUVEAU,
                        "assemblage_base": mb.get("assemblage")}
    if any("self_attn" in n for n in pris):
        mb["attn_int8"] = md.get("attn_int8", mb.get("attn_int8"))
    with open(os.path.join(a.sortie, "acvram_manifest.json"), "w") as f:
        json.dump(mb, f, ensure_ascii=False, indent=1)
    for fn in sorted(os.listdir(a.base)):
        if fn == "acvram_manifest.json":
            continue
        cible = os.path.join(a.sortie, fn)
        if not os.path.exists(cible):
            os.symlink(os.path.relpath(os.path.join(a.base, fn), a.sortie), cible)
    # contrôle : chaque clé du weight_map est lisible dans son fragment
    absents = []
    lus: dict[str, set] = {}
    for k, fn in wm.items():
        if fn not in lus:
            with safe_open(os.path.join(a.sortie, fn), framework="pt", device="cpu") as fh:
                lus[fn] = set(fh.keys())
        if k not in lus[fn]:
            absents.append(k)
    octets = sum(t.numel() * t.element_size() for t in tenseurs.values())
    print(json.dumps({"sortie": os.path.basename(os.path.normpath(a.sortie)), "tenseurs_pris": len(pris), "cles": len(tenseurs),
                      "mo_ecrits": round(octets / 1e6, 1), "cles_absentes": len(absents), "exemple": pris[0]}, ensure_ascii=False))
    return 1 if absents else 0


if __name__ == "__main__":
    raise SystemExit(main())
