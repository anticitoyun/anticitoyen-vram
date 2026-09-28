"""Pièce 241 : produit les noyaux PRÉCOMPILÉS du Flatpak (236/240) dans un conteneur CUDA sans carte.

    python -m acvram.kernels.precompiles --dossier build/noyaux --archs 12.0,8.6

Compile `acvram_kernels.cu` pour chaque architecture demandée (une seule compilation, plusieurs `-gencode`) avec nvcc,
sans GPU, et range `acvram_kernels.so` + `empreinte.json` sous ``<dossier>/<src_sha16>/`` — exactement ce que
`kernels._precompile_utilisable` relit au chargement (même écriture : `_ecrire_empreinte`). La CI joue ceci dans
`nvidia/cuda:*-devel` avec la roue torch cu130 ; le manifeste Flathub embarque ``build/noyaux`` sous
``/app/lib/acvram/noyaux`` (`ACVRAM_KERNELS_PRECOMPILES`). 7x8b : le port Marlin aussi, sous ``<dossier>/marlin-<empreinte>/``
(`marlin_port.compiler_precompile`, relu par `marlin_port.precompile_utilisable`)."""
from __future__ import annotations

import argparse
import json
import os
import sys

DOSSIER_DEFAUT = "build/noyaux"          # = `path: ../../build/noyaux` du manifeste (tests/test_noyaux_precompiles_ci_241.py)
ARCHS_DEFAUT = "12.0"                    # RTX 5090 (sm_120f) ; ajouter 8.9,8.6 pour Ada/Ampere


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dossier", default=DOSSIER_DEFAUT)
    ap.add_argument("--archs", default=ARCHS_DEFAUT, help="MAJEUR.MINEUR séparés par des virgules, ex. 12.0,8.6")
    a = ap.parse_args(argv)
    from acvram import kernels
    archs = kernels.archs_depuis_texte(a.archs)
    if not archs:
        print("aucune architecture", file=sys.stderr)
        return 2
    cand = kernels.compiler_precompile(a.dossier, archs)
    with open(os.path.join(cand, "empreinte.json"), encoding="utf-8") as fh:
        e = json.load(fh)
    # 7x8b : Marlin aussi — sans lui, le Flatpak servait tous les MoE nvfp4 sur la pile naturelle
    from acvram.kernels import marlin_port
    cand_m = marlin_port.compiler_precompile(a.dossier, archs)
    print(json.dumps({"dossier": cand, "archs": e["archs"], "src_hash": e["src_hash"], "torch": e["torch"], "cuda": e["cuda"],
                      "so_octets": os.path.getsize(os.path.join(cand, "acvram_kernels.so")),
                      "marlin_dossier": cand_m, "marlin_so_octets": os.path.getsize(os.path.join(cand_m, "acvram_marlin.so"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
