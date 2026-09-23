#!/usr/bin/env python3
"""
Intègre les PPL et débits depuis les verdicts dans claude-modeles.md et kimi-modeles.md.
Format : "PPL X,XXXX juge (méd X,XXXX) — verdict-source"
Source : sage-menus-mise-a-jour-partielle-17-09.md

Usage :
  python integrer-verdicts-menus-17-09.py
"""

import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

# Mappages modèle → (fichier_verdict, clés_ppl, clés_débit_b1)
MODELES = {
    "GLM-4.7-Flash-nvfp4": {
        "verdict": "sage-calibration-verdict-17-09.md",
        "ppl_key": "1,015 géo",  # depuis sage-calibration-verdict
        "ppl_med": "1,024",
        "b1_key": None,
    },
    "GLM-4.7-Flash-srcbf16-nvfp4": {
        "verdict": "sage-calibration-verdict-17-09.md",
        "ppl_key": "1,015 géo",
        "ppl_med": "1,024",
        "b1_key": None,
    },
    "Coder-30B-A3B": {
        "verdict": "sage-calibration-verdict-17-09.md",
        "ppl_key": "1,027 géo",  # Coder (depuis sage-calibration-verdict)
        "ppl_med": None,
        "b1_key": None,
    },
    "Qwen3.8-27B-nvfp4": {
        "verdict": "verdict-qwen38-calibA-17-09.md",
        "ppl_key": "1,0253 géo",
        "ppl_med": "1,0152",
        "b1_key": "63,8 t/s",
    },
    "Kimi-Linear-35B-kda-nvfp4": {
        "verdict": "verdict-palier2-kimi-linear-17-09.md",
        "ppl_key": None,  # Non classée
        "ppl_med": None,
        "b1_key": "218,4 t/s",
    },
    "Gemma-4-26B-A4B": {
        "verdict": "verdict-palier2-gemma-4-26b-17-09.md",
        "ppl_key": None,  # Non classable
        "ppl_med": None,
        "b1_key": "228,2 t/s",
    },
    "Nemotron-3.5-Lightning-30B-A3B-NVFP4": {
        "verdict": "verdict-palier2-nemotron-17-09.md",
        "ppl_key": "0,987 géo",  # vLLM
        "ppl_med": "0,993",
        "b1_key": "401,1 t/s",
    },
}

def format_ppl_line(ppl_key, ppl_med, verdict_file):
    """Formate une ligne PPL avec juge et source."""
    if not ppl_key:
        return "PPL non classée"
    parts = [f"PPL {ppl_key}"]
    if ppl_med:
        parts[-1] += f" (méd {ppl_med})"
    parts.append(f"— {verdict_file}")
    return " ".join(parts)

def main():
    # Vérifier que les fichiers verdicts existent
    for info in MODELES.values():
        verdict_path = REVUE / info["verdict"]
        if not verdict_path.exists():
            print(f"ERREUR: Fichier verdict manquant: {verdict_path}")
            sys.exit(1)

    # Afficher résumé des verdicts à intégrer
    print("=== Verdicts à intégrer ===\n")
    for model, info in MODELES.items():
        ppl_line = format_ppl_line(info["ppl_key"], info["ppl_med"], info["verdict"])
        b1_line = f"b=1: {info['b1_key']}" if info["b1_key"] else "b=1: —"
        print(f"{model}:")
        print(f"  {ppl_line}")
        print(f"  {b1_line}")
        print()

    print("Script de préparation terminé.")
    print("Étapes suivantes:")
    print("1. Mettre à jour claude-modeles.md manuellement ou via script")
    print("2. Mettre à jour kimi-modeles.md")
    print("3. Ajouter test de vérification dans test_menus.py")

if __name__ == "__main__":
    main()
