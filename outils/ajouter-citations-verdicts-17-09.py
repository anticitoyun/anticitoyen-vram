#!/usr/bin/env python3
"""
Ajoute les citations de verdicts aux débits b=1 déjà présents dans les menus.
Utilise le plan TSV pour déduire les blocs des modèles.

Modification : "| b=1 XXX,X t/s" → "| b=1 XXX,X t/s — verdict-palier1-bloc<N>-17-09"
"""

import csv
import re
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"
PLAN_PATH = Path("/tmp/palier1-17-09-plan.tsv")

def load_plan():
    """Charge le plan TSV pour obtenir les blocs."""
    plan = {}
    with open(PLAN_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            nom = row['nom'].strip()
            bloc = row['bloc'].strip()
            plan[nom] = int(bloc)
    return plan

def add_citation_to_line(line, model_name, bloc):
    """Ajoute la citation de verdict si le débit b=1 est présent."""

    # Chercher le pattern "| b=1 XXX,X t/s" (sans citation)
    pattern = r'\|\s*b=1\s+([\d,\.]+\s+t/s)(?!\s*—)'

    if not re.search(pattern, line):
        return None  # Pas de b=1 sans citation

    # Remplacer par le même débit avec citation
    citation = f"verdict-palier1-bloc{bloc}-17-09"
    new_line = re.sub(pattern, f'| b=1 \\1 — {citation}', line)

    return new_line if new_line != line else None

def main():
    print("Chargement du plan...")
    plan = load_plan()
    print(f"Plan chargé : {len(plan)} modèles")

    menu_files = [
        ("claude", REVUE / "claude-modeles.md"),
        ("kimi", REVUE / "kimi-modeles.md"),
    ]

    total_updated = 0

    for menu_name, menu_path in menu_files:
        if not menu_path.exists():
            print(f"Attention : {menu_path.name} introuvable")
            continue

        with open(menu_path) as f:
            lines = f.readlines()

        new_lines = []
        menu_updated = 0

        for line in lines:
            updated_line = None
            # Chercher un modèle du plan dans cette ligne
            for model_name in plan:
                if f"`{model_name}`" in line:
                    bloc = plan[model_name]
                    updated_line = add_citation_to_line(line.rstrip(), model_name, bloc)
                    if updated_line:
                        menu_updated += 1
                        break

            if updated_line:
                new_lines.append(updated_line + '\n')
            else:
                new_lines.append(line)

        if menu_updated > 0:
            with open(menu_path, 'w') as f:
                f.writelines(new_lines)
            print(f"OK {menu_path.name} : {menu_updated} citations ajoutées")

        total_updated += menu_updated

    print(f"\nTotal : {total_updated} citations ajoutées")

if __name__ == "__main__":
    main()
