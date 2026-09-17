#!/usr/bin/env python3
"""
Joindre débits extraits avec plan TSV pour obtenir noms complets et débits.
Parser manuel du CSV pour gérer les virgules dans les débits.
"""

import csv
import re
from pathlib import Path

PLAN_PATH = Path("/tmp/palier1-17-09-plan.tsv")
DEBITS_CSV = Path("/tmp/palier1-b1-mapping.csv")
OUT_CSV = Path("/tmp/palier1-menus-a-enrichir.csv")

def load_plan():
    """Charge le plan TSV."""
    plan = {}
    with open(PLAN_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            bloc = int(row['bloc'])
            rang = int(row['rang'])
            nom = row['nom'].strip()
            if bloc not in plan:
                plan[bloc] = {}
            plan[bloc][rang] = nom
    return plan

def load_debits():
    """Charge les débits extraits avec parsing manuel."""
    debits_by_bloc = {}
    with open(DEBITS_CSV) as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skip header
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')

        # Chercher où la colonne bloc_source commence (verdict-palier...)
        verdict_idx = next((i for i, p in enumerate(parts) if 'verdict' in p), None)
        if not verdict_idx:
            continue

        model = parts[0]
        debit = ','.join(parts[1:verdict_idx])
        bloc_file = ','.join(parts[verdict_idx:])

        # Extraire le bloc
        match = re.search(r'bloc(\d+)', bloc_file)
        if match:
            bloc = int(match.group(1))
            if bloc not in debits_by_bloc:
                debits_by_bloc[bloc] = []
            debits_by_bloc[bloc].append((model, debit, bloc_file))

    return debits_by_bloc

def main():
    print("Chargement du plan...")
    plan = load_plan()
    print(f"Plan charge : {sum(len(b) for b in plan.values())} modeles")

    print("Chargement des debits extraits...")
    debits = load_debits()
    total_debits = sum(len(b) for b in debits.values())
    print(f"Debits charges : {total_debits} modeles")

    # Creer la table de jointure
    results = []

    for bloc in sorted(debits.keys()):
        for model_abrege, debit, bloc_file in debits[bloc]:
            # Chercher le nom complet dans le plan
            found = False
            for rang, nom_complet in plan.get(bloc, {}).items():
                # Chercher une correspondance (substring match)
                if model_abrege.lower() in nom_complet.lower():
                    results.append({
                        'nom_complet': nom_complet,
                        'nom_abrege': model_abrege,
                        'debit_b1': debit,
                        'bloc_source': bloc_file,
                        'rang': rang,
                        'bloc': bloc,
                    })
                    print(f"OK: {model_abrege:30} -> {nom_complet}")
                    found = True
                    break

            if not found:
                print(f"WARN: Pas de correspondance pour {model_abrege} (bloc {bloc})")

    # Afficher en CSV
    print(f"\nResultats : {len(results)} jointures")
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rang', 'bloc', 'nom_complet', 'debit_b1', 'bloc_source'])
        writer.writeheader()
        for row in sorted(results, key=lambda x: (x['bloc'], x['rang'])):
            writer.writerow({
                'rang': row['rang'],
                'bloc': row['bloc'],
                'nom_complet': row['nom_complet'],
                'debit_b1': row['debit_b1'],
                'bloc_source': row['bloc_source'],
            })

    print(f"Ecrit : {OUT_CSV}")

if __name__ == "__main__":
    main()
