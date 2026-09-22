#!/usr/bin/env python3
"""
Propose un CSV de structure pour les 142 modèles palier 1.
Lit les verdicts bloc par bloc et propose les modèles trouvés (format libre).

Sortie : CSV partiellement rempli que l'utilisateur complète avec les PPL/débits.
Utilisation :
  python proposer-palier1-csv-17-09.py > palier1-template-17-09.csv
  # L'utilisateur remplit les colonnes PPL, juge, débit_b1, etc.
  python integrer-palier1-verdicts-17-09.py palier1-template-17-09.csv --dry-run
"""

import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

# Expressions régulières pour extraire les modèles du texte narratif
MODEL_PATTERNS = [
    # Modèles nommés avec backticks ou guillemets
    r'`([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm))`',
    # Modèles en gras dans le texte
    r'\*\*([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm))\*\*',
    # Modèles énumérés avec tirets
    r'[·\-]\s+([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm))',
    # Modèles avec débit (pattern : "Nom-Model t/s" ou "Nom-Model j/s")
    r'([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm))\s+(?:\*\*)?\d+[\.,]\d+(?:\s+(?:t/s|j/s))?',
]

def extract_models_from_verdict(verdict_path):
    """Extrait les noms de modèles d'un fichier verdict."""
    models_found = set()

    with open(verdict_path) as f:
        content = f.read()

    # Chercher tous les noms de modèles mentionnés
    for pattern in MODEL_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            model_name = match.group(1).strip()
            if len(model_name) > 3 and not model_name.startswith('_'):
                models_found.add(model_name)

    return sorted(models_found)

def main():
    print("modèle,format,PPL,juge,PPL_médiane,débit_b1,régime,bloc_source", file=sys.stdout)

    total_models = 0
    for bloc_num in range(8):
        verdict_path = REVUE / f"verdict-palier1-bloc{bloc_num}-17-09.md"
        if not verdict_path.exists():
            print(f"ERREUR : {verdict_path.name} introuvable", file=sys.stderr)
            continue

        models = extract_models_from_verdict(verdict_path)
        bloc_name = f"verdict-palier1-bloc{bloc_num}-17-09"

        print(f"# Bloc {bloc_num} : {len(models)} modèles trouvés", file=sys.stderr)

        for model in models:
            # Déduire le format du nom du modèle
            format_guess = ""
            if "nvfp4" in model:
                format_guess = "nvfp4"
            elif "bf16" in model:
                format_guess = "bf16"
            elif "int8" in model:
                format_guess = "int8"
            elif "gguf" in model:
                format_guess = "GGUF"
            elif "exl3" in model:
                format_guess = "EXL3"
            elif "vllm" in model:
                format_guess = "vLLM"

            # CSV : modèle, format (colonnes remplies),
            # PPL, juge, PPL_médiane, débit_b1, régime (à remplir)
            print(
                f"{model},{format_guess},,,,,,{bloc_name}",
                file=sys.stdout
            )
            total_models += 1

    print(f"\nTotal : {total_models} modèles proposés", file=sys.stderr)
    print("\nINSTRUCTIONS :", file=sys.stderr)
    print("1. Relire le verdict correspondant pour chaque bloc", file=sys.stderr)
    print("2. Compléter les colonnes : PPL, juge, PPL_médiane, débit_b1, régime", file=sys.stderr)
    print("3. Lancer : python integrer-palier1-verdicts-17-09.py <fichier.csv> --dry-run", file=sys.stderr)
    print("4. Vérifier les changements avec : git diff claude-modeles.md", file=sys.stderr)

if __name__ == "__main__":
    main()
