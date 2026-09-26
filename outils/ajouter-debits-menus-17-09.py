#!/usr/bin/env python3
"""
Ajoute les débits b=1 dans claude-modeles.md et kimi-modeles.md.
Prend un CSV : modèle, débit_b1, bloc_source

Usage :
  python3 ajouter-debits-menus-17-09.py /tmp/palier1-b1-mapping.csv --dry-run
  python3 ajouter-debits-menus-17-09.py /tmp/palier1-b1-mapping.csv  # réel
"""

import csv
import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

def load_mapping(csv_path):
    """Charge le CSV modèle → débit."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['modèle'].strip()
            mapping[model] = {
                'debit_b1': row['débit_b1'].strip(),
                'bloc_source': row['bloc_source'].strip(),
            }
    return mapping

def add_debit_to_menu(line, model_name, mapping):
    """
    Ajoute le débit b=1 à une ligne de menu.
    Cherche backtick + modèle + paramètres.
    Ajoute avant le premier em-dash ou en fin de ligne.
    """
    if model_name not in mapping:
        return None  # Pas dans le mapping

    m = mapping[model_name]
    debit_info = f"| b=1 {m['debit_b1']} — {m['bloc_source']}"

    # Chercher le modèle dans la ligne
    pattern = rf'`{re.escape(model_name)}`'
    if not re.search(pattern, line):
        return None

    # Chercher où insérer (après la taille 'XXG', avant le em-dash ou en fin)
    # Format : `Model` (17G) — ...

    # Si la ligne contient déjà "| b=1", ne pas dupliquer
    if "| b=1" in line:
        return None

    # Trouver la position après la taille
    match = re.match(r'(.+?\(\d+[A-Za-z]\))\s*(.*)', line)
    if not match:
        return None

    prefix = match.group(1)
    suffix = match.group(2)

    # Ajouter avant le suffix (ou en fin si vide)
    if suffix.startswith('—'):
        new_line = f"{prefix} — {debit_info} {suffix[1:].strip()}"
    elif suffix:
        new_line = f"{prefix} — {debit_info} {suffix}"
    else:
        new_line = f"{prefix} — {debit_info}"

    return new_line

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ajouter-debits-menus-17-09.py <mapping.csv> [--dry-run]", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if not csv_path.exists():
        print(f"Erreur : {csv_path} introuvable", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(csv_path)
    print(f"Chargé : {len(mapping)} modèles du mapping", file=sys.stderr)

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
            # Chercher le modèle dans cette ligne
            for model_name in mapping:
                if f"`{model_name}`" in line:
                    updated_line = add_debit_to_menu(line.rstrip(), model_name, mapping)
                    if updated_line:
                        print(f"{menu_name} : {model_name} → +{mapping[model_name]['debit_b1']}", file=sys.stderr)
                        menu_updated += 1
                        break

            if updated_line:
                new_lines.append(updated_line + '\n')
            else:
                new_lines.append(line)

        if not dry_run and menu_updated > 0:
            with open(menu_path, 'w') as f:
                f.writelines(new_lines)
            print(f"✓ {menu_path.name} : {menu_updated} lignes mises à jour", file=sys.stderr)
        elif dry_run and menu_updated > 0:
            print(f"[DRY-RUN] {menu_path.name} : {menu_updated} lignes prêtes à être mises à jour", file=sys.stderr)

        total_updated += menu_updated

    print(f"\nTotal : {total_updated} modèles enrichis", file=sys.stderr)
    if dry_run:
        print("Mode dry-run : aucun fichier modifié", file=sys.stderr)

if __name__ == "__main__":
    main()
