#!/usr/bin/env python3
"""
Intègre les PPL + débits du palier 1 dans claude-modeles.md et kimi-modeles.md.
Prend un CSV de mapping (modèle, PPL, juge, PPL_médiane, débit_b1, régime, verdict_file).

Usage :
  python extraire-palier1-verdicts-17-09.py > /tmp/palier1-mapping.csv
  python integrer-palier1-verdicts-17-09.py /tmp/palier1-mapping.csv --dry-run

Format attendu dans menus avant intégration :
  - `Modèle-X` (17G) — description simple

Format attendu après intégration :
  - `Modèle-X` (17G) — PPL 0,987 géo (méd 0,993) — verdict-palier1-bloc0-17-09 | b=1 123,4 t/s — verdict-palier1-bloc0-17-09 | description simple
"""

import csv
import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

def load_mapping(csv_path):
    """Charge le CSV de mapping modèle → métriques."""
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['modèle'].strip()
            mapping[model] = {
                'ppl': row['PPL'].strip(),
                'juge': row['juge'].strip(),
                'ppl_med': row['PPL_médiane'].strip(),
                'debit_b1': row['débit_b1'].strip(),
                'regime': row['régime'].strip(),
                'verdict_file': row['fichier_verdict'].strip().replace('.md', ''),
            }
    return mapping

def format_ppl_line(m):
    """Formate une ligne PPL avec juge et source."""
    if not m['ppl']:
        return None

    parts = [f"PPL {m['ppl']} {m['juge']}"]
    if m['ppl_med']:
        parts[0] += f" (méd {m['ppl_med']})"
    parts.append(f"— {m['verdict_file']}")

    return " ".join(parts)

def format_debit_line(m):
    """Formate une ligne débit b=1 avec source."""
    if not m['debit_b1']:
        return None

    return f"| b=1 {m['debit_b1']} — {m['verdict_file']}"

def integrate_model(line, model_name, mapping):
    """
    Intègre les métriques pour un modèle dans une ligne de menu.
    Retourne la ligne modifiée, ou None si le modèle ne figure pas dans le mapping.

    Cherche `model_name` entre backticks ou à la fin d'une ligne.
    Ajoute les métriques avant le premier em-dash ou à la fin de la ligne.
    """
    if model_name not in mapping:
        return None  # Modèle non trouvé dans le mapping

    m = mapping[model_name]
    ppl_line = format_ppl_line(m)
    debit_line = format_debit_line(m)

    if not ppl_line:
        return None  # Pas de PPL à ajouter

    # Chercher la ligne avec ce modèle (backticks ou direct)
    pattern = rf'`{re.escape(model_name)}`|^[^(]*{re.escape(model_name)}'
    if not re.search(pattern, line):
        return None  # Modèle non trouvé dans la ligne

    # Chercher l'endroit où insérer (après la taille, avant le commentaire)
    # Format : `Model` (17G) — [new metrics] | [old comment]
    match = re.match(r'(.+?)\([\d\.]+[A-Za-z]\)(.*)$', line)
    if not match:
        return None

    prefix = match.group(1)
    suffix = match.group(2)

    # Construire la nouvelle ligne
    metrics = ppl_line
    if debit_line:
        metrics += f" {debit_line}"

    # Ajouter le suffix (ancienne description) s'il existe
    if suffix.strip():
        # Éviter les em-dashes dupliqués
        if suffix.lstrip().startswith('—'):
            new_line = f"{prefix}({m['regime']}) — {metrics} {suffix}"
        else:
            new_line = f"{prefix}({m['regime']}) — {metrics} {suffix}"
    else:
        new_line = f"{prefix}({m['regime']}) — {metrics}"

    return new_line

def main():
    if len(sys.argv) < 2:
        print("Usage: python integrer-palier1-verdicts-17-09.py <mapping.csv> [--dry-run]", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if not csv_path.exists():
        print(f"Erreur : {csv_path} introuvable", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(csv_path)
    print(f"Chargé : {len(mapping)} modèles du mapping", file=sys.stderr)

    # Parcourir les menus et afficher les changements
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
            # Chercher tous les modèles du mapping dans cette ligne
            for model_name in mapping:
                if f"`{model_name}`" in line or line.endswith(model_name):
                    updated_line = integrate_model(line.rstrip(), model_name, mapping)
                    if updated_line:
                        print(f"{menu_name} : {model_name} → intégré", file=sys.stderr)
                        menu_updated += 1
                        break

            if updated_line:
                new_lines.append(updated_line + '\n')
            else:
                new_lines.append(line)

        total_updated += menu_updated

        if not dry_run and menu_updated > 0:
            with open(menu_path, 'w') as f:
                f.writelines(new_lines)
            print(f"✓ {menu_path.name} : {menu_updated} modèles mis à jour", file=sys.stderr)
        elif dry_run:
            print(f"[DRY-RUN] {menu_path.name} : {menu_updated} modèles prêts à être mis à jour", file=sys.stderr)

    print(f"\nTotal : {total_updated} modèles intégrés", file=sys.stderr)
    if dry_run:
        print("Mode dry-run : aucun fichier modifié", file=sys.stderr)

if __name__ == "__main__":
    main()
