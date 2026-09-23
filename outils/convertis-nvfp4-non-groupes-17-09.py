#!/usr/bin/env python3
"""Convertis dont le manifeste porte au moins une projection NVFP4 HORS experts
routés — celles que `nvfp4_matmul` sert au prefill, donc en W8A8 tacite avant
le 17/09 (sage-prefill-a8-verdict-17-09 § 2 : ces PPL se réétiquettent
« prefill w8a8 »). Un converti sans ces projections (experts NVFP4 groupés,
reste int8/bf16) n'est pas concerné.

    python outils/convertis-nvfp4-non-groupes-17-09.py <dossier-parc> [...]
"""
import collections
import json
import os
import re
import sys

for parc in sys.argv[1:]:
    for nom in sorted(os.listdir(parc)):
        f = os.path.join(parc, nom, "acvram_manifest.json")
        if not os.path.isfile(f):
            continue
        try:
            m = json.load(open(f))
        except (OSError, ValueError) as e:
            print(f"{nom:52s} manifeste illisible ({type(e).__name__})")
            continue
        genres = collections.Counter()
        for k, v in m.get("tensors", {}).items():
            if v.get("format") != "nvfp4" or ".experts." in k:
                continue
            g = re.sub(r"layers\.\d+\.", "layers.N.", k).replace("model.layers.N.", "")
            genres[g.replace(".weight", "")] += 1
        if genres:
            print(f"{nom:52s} W8A8-TACITE  {sum(genres.values()):4d} proj.  "
                  + ", ".join(f"{g}×{n}" for g, n in sorted(genres.items())))
        else:
            print(f"{nom:52s} non concerné")
