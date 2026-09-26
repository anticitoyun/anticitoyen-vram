#!/usr/bin/env python3
"""
Liste TOUS les noms de modèles mentionnés dans les verdicts palier 1.
Le mapping des débits est complété manuellement depuis les verdicts.

Usage :
  python3 lister-modeles-palier1-17-09.py
"""

import re
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

# Chercher des mots qui ressemblent à des noms de modèles
# Pattern : mots avec traits d'union, chiffres, minuscules
# Exclure les mots trop courts ou les noms génériques
MODEL_PATTERN = r'\b([A-Za-z][A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm|Q[0-9]\w+|heretic|Q6_K|Q4_K_M|Q5_K_M|bf16|alphacommun|temoin|srcgguf|srcQ|srcexl|src\w+|purepur|\w+pure))\b'

def extract_models_from_verdict(verdict_path):
    """Extrait tous les noms de modèles mentionnés dans un verdict."""
    models = set()

    with open(verdict_path) as f:
        content = f.read()

    # Chercher tous les candidats
    for match in re.finditer(MODEL_PATTERN, content):
        name = match.group(1)

        # Filtrer les faux positifs
        if len(name) < 3:
            continue
        if name.lower() in ('bloc', 'csv', 'json', 'gpg', 'gpu', 'cpu', 'log', 'tsr', 'api'):
            continue
        if name.startswith(('_', '.')):
            continue

        models.add(name)

    return sorted(models)

def main():
    print("# Modèles palier 1 — Extraction depuis verdicts\n")
    print("Copier-coller les débits b=1 depuis les sections 'mesuré' de chaque verdict.\n")

    total = 0
    for bloc_num in range(8):
        bloc_path = REVUE / f"verdict-palier1-bloc{bloc_num}-17-09.md"
        if not bloc_path.exists():
            print(f"⚠ {bloc_path.name} introuvable")
            continue

        models = extract_models_from_verdict(bloc_path)
        print(f"## Bloc {bloc_num} : {len(models)} candidats")
        print(f"Fichier : `{bloc_path.name}`\n")
        print("| Modèle | Débit b=1 | Refus |")
        print("|--------|-----------|-------|")

        for model in models:
            print(f"| `{model}` | | |")

        print()
        total += len(models)

    print(f"**Total : {total} modèles listés**\n")
    print("Instructions :")
    print("1. Ouvrir chaque bloc verdict dans l'éditeur")
    print("2. Copier-coller les débits b=1 depuis la section 'mesuré'")
    print("3. Remplir le tableau ci-dessus (ou exporter en CSV)")

if __name__ == "__main__":
    main()
