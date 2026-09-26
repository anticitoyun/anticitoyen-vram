#!/usr/bin/env python3
"""
Applique les débits b=1 aux menus depuis le CSV jointé (nom_complet, débit_b1).
"""

import csv
import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

def load_mapping(csv_path):
    """Charge le CSV jointé."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_complet = row['nom_complet'].strip()
            mapping[model_complet] = {
                'debit_b1': row['debit_b1'].strip(),
                'bloc_source': row['bloc_source'].strip(),
            }
    return mapping

def add_debit_to_line(line, model_name, mapping):
    """Ajoute le débit à une ligne si elle contient le modèle."""
    if model_name not in mapping:
        return None

    m = mapping[model_name]

    # Chercher le modèle dans la ligne (entre backticks)
    if f"`{model_name}`" not in line:
        return None

    # Si la ligne a déjà "| b=1", ne pas dupliquer
    if "| b=1" in line:
        return None

    # Chercher la position après la taille
    match = re.match(r'(.+?\([\d\.]+[A-Za-z]\))\s*(.*)', line)
    if not match:
        return None

    prefix = match.group(1)
    suffix = match.group(2)

    # Ajouter le débit
    debit_info = f"| b=1 {m['debit_b1']} — {m['bloc_source']}"

    if suffix.startswith('—'):
        new_line = f"{prefix} — {debit_info} {suffix[1:].strip()}"
    elif suffix:
        new_line = f"{prefix} — {debit_info} {suffix}"
    else:
        new_line = f"{prefix} — {debit_info}"

    return new_line

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 appliquer-debits-menus-17-09.py <mapping.csv> [--dry-run]", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if not csv_path.exists():
        print(f"Erreur : {csv_path} introuvable", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(csv_path)
    print(f"Charge : {len(mapping)} modeles", file=sys.stderr)

    menu_files = [
        ("claude", REVUE / "claude-modeles.md"),
        ("kimi", REVUE / "kimi-modeles.md"),
    ]

    total_updated = 0

    for menu_name, menu_path in menu_files:
        if not menu_path.exists():
            print(f"Attention : {menu_path.name} introuvable", file=sys.stderr)
            continue

        with open(menu_path) as f:
            lines = f.readlines()

        new_lines = []
        menu_updated = 0

        for line in lines:
            updated_line = None
            # Chercher le modele dans cette ligne
            for model_name in mapping:
                if f"`{model_name}`" in line:
                    updated_line = add_debit_to_line(line.rstrip(), model_name, mapping)
                    if updated_line:
                        print(f"{menu_name} : {model_name} +debit", file=sys.stderr)
                        menu_updated += 1
                        break

            if updated_line:
                new_lines.append(updated_line + '\n')
            else:
                new_lines.append(line)

        if not dry_run and menu_updated > 0:
            with open(menu_path, 'w') as f:
                f.writelines(new_lines)
            print(f"OK {menu_path.name} : {menu_updated} lignes mises a jour", file=sys.stderr)
        elif dry_run and menu_updated > 0:
            print(f"[DRY-RUN] {menu_path.name} : {menu_updated} lignes pret", file=sys.stderr)

        total_updated += menu_updated

    print(f"\nTotal : {total_updated} modeles enrichis", file=sys.stderr)
    if dry_run:
        print("Mode dry-run : aucun fichier modifie", file=sys.stderr)

if __name__ == "__main__":
    main()
