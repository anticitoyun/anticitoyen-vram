#!/usr/bin/env python3
"""
Extrait précisément les modèles et débits b=1 depuis la section "b=1 :" des verdicts.

Format dans les verdicts :
  b=1 : Qwen3-4B 188,0 · Qwen3.5-4B-heretic 238,4 · A1-4B 212,6 / 214,2 ...

Génère un CSV : modèle,débit_b1,bloc_source

Usage :
  python3 extraire-b1-verdicts-17-09.py > /tmp/palier1-b1-17-09.csv
"""

import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

def parse_b1_section(b1_text):
    """Parse une section "b=1 : ..." et retourne liste de (modèle, débit)."""
    models = {}

    # Splitter par les énumérateurs (·, ;, /  pour les alternatives)
    # Mais attention : "/" peut être dans une alternative (A1-4B 212,6 / 214,2)
    # Donc d'abord splitter par ";" pour la fin de la section b=1

    b1_section = b1_text.split(';')[0] if ';' in b1_text else b1_text

    # Splitter par "·" pour chaque modèle
    parts = b1_section.split('·')

    for part in parts:
        part = part.strip()
        if not part or len(part) < 3:
            continue

        # Parser "NomModèle débit" ou "NomModèle débits alternatifs"
        # Patterns :
        # - "Qwen3-4B 188,0"
        # - "A1-4B 212,6 / 214,2 (témoin = converti, ...)"
        # - "Llama-2-7b 153,0 t/s"
        # - "phi-4 Q4_K_M 139,9" (modèle + format)

        # Chercher le premier nombre (qui est le débit)
        match = re.match(r'([A-Za-z0-9\-\. ]+?)\s+([\d,\.]+)', part)
        if match:
            model_name = match.group(1).strip()
            debit = match.group(2).replace('.', ',')  # Normaliser en virgule française

            # Nettoyer le nom du modèle (retirer les formats)
            model_name = re.sub(r'\s+(Q\d_K|int\d|bf\d|src\w+|nvfp4|gguf)', '', model_name).strip()

            if model_name and len(model_name) > 2:
                # Garder le premier débit (en cas d'alternatives avec /)
                if model_name not in models:
                    models[model_name] = f"{debit} t/s"

    return models

def extract_from_bloc(verdict_path):
    """Extrait les modèles b=1 d'un bloc verdict."""
    with open(verdict_path) as f:
        content = f.read()

    # Chercher la section "b=1 :" jusqu'au prochain ";"
    match = re.search(r'b=1\s*:\s*([^;]+?)(?:;|$)', content)
    if not match:
        return {}

    b1_section = match.group(1)
    return parse_b1_section(b1_section)

def main():
    all_models = {}

    print("modèle,débit_b1,bloc_source", file=sys.stdout)

    for bloc_num in range(8):
        bloc_path = REVUE / f"verdict-palier1-bloc{bloc_num}-17-09.md"
        if not bloc_path.exists():
            print(f"⚠ {bloc_path.name} introuvable", file=sys.stderr)
            continue

        models = extract_from_bloc(bloc_path)
        bloc_name = f"verdict-palier1-bloc{bloc_num}-17-09"

        for model_name, debit in sorted(models.items()):
            if model_name not in all_models:
                all_models[model_name] = (debit, bloc_name)
                print(f"{model_name},{debit},{bloc_name}.md", file=sys.stdout)

        print(f"Bloc {bloc_num} : {len(models)} modèles extraits", file=sys.stderr)

    print(f"\nTotal : {len(all_models)} modèles", file=sys.stderr)

if __name__ == "__main__":
    main()
