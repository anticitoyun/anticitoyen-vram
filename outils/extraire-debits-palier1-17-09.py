#!/usr/bin/env python3
"""
Extrait les débits b=1 depuis les 8 blocs verdict palier 1.
Le palier 1 ne contient pas de PPL (non classables), juste débits b=1 et refus.

Génère un CSV : modèle → (débit_b1, bloc_source)

Usage :
  python3 extraire-debits-palier1-17-09.py > /tmp/palier1-debits-17-09.csv
"""

import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

# Patterns pour extraire débit + modèle
# Format : "Modèle-Nom débit" ou "Modèle-Nom **débit**"
DEBIT_PATTERNS = [
    # Modèle suivi de débit en gras
    r'(\*\*)?([A-Za-z0-9\-\.]+)\s+(\*\*)?([\d,\.]+\s+(?:t/s|j/s))(\*\*)?',
    # Modèle suivi de débit simple avec unité
    r'([A-Za-z0-9\-\.]+)\s+([0-9,\.]+)\s+t/s',
]

def parse_bloc_verdict_debits(bloc_path):
    """Parse un bloc verdict et extrait modèles → débits."""
    results = []
    verdict_name = bloc_path.stem

    with open(bloc_path) as f:
        content = f.read()

    # Chercher la section "mesuré" qui contient les débits
    # Format : "b=1 : Modèle1 123,4 · Modèle2 234,5 t/s ; ..."
    # Ou parfois : "Modèle **123,4** t/s"

    # Extraire la section b=1
    b1_match = re.search(r'b=1\s*:([^;]+?)(?:;|refus:|verdict:)', content, re.DOTALL)
    if not b1_match:
        return results

    b1_section = b1_match.group(1)

    # Splitter par · ou / pour les modèles énumérés
    parts = re.split(r'[·/\s]{1,3}', b1_section)

    models_found = {}
    for part in parts:
        part = part.strip()
        if not part or len(part) < 3:
            continue

        # Parser "ModèleName débit"
        # Patterns : "Qwen3-4B 188,0", "A1-4B 212,6 / 214,2", "Llama-2-7b 153,0 t/s"
        match = re.match(r'([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm)?)\s+(\d+[,\.]\d+)\s*(?:t/s)?', part)
        if match:
            model_name = match.group(1)
            debit = match.group(2)

            # Normaliser : virgule → virgule (garder format français)
            debit = debit.replace('.', ',')

            # Garder le debit en t/s (pas j/s, sinon c'est prefill)
            if model_name not in models_found:
                models_found[model_name] = f"{debit} t/s"

    # Ajouter aussi les modèles mentionnés avec leurs chiffres en gras
    # Format : "Modèle **123,4** t/s"
    bold_pattern = r'([A-Za-z0-9\-\.]+(?:nvfp4|bf16|int8|gguf|exl3|vllm)?)\s+\*\*(\d+[,\.]\d+)\s*(?:t/s)?\*\*'
    for match in re.finditer(bold_pattern, b1_section):
        model_name = match.group(1)
        debit = match.group(2).replace('.', ',')
        if model_name not in models_found:
            models_found[model_name] = f"{debit} t/s"

    for model_name, debit in models_found.items():
        results.append({
            'model': model_name,
            'debit_b1': debit,
            'bloc': verdict_name,
        })

    return results

def main():
    # Lire tous les blocs
    all_models = {}  # modèle -> (débit, bloc) pour dédupliquer

    for bloc_num in range(8):
        bloc_path = REVUE / f"verdict-palier1-bloc{bloc_num}-17-09.md"
        if not bloc_path.exists():
            print(f"Attention : {bloc_path.name} introuvable", file=sys.stderr)
            continue

        models = parse_bloc_verdict_debits(bloc_path)
        print(f"Bloc {bloc_num} : {len(models)} modèles extraits", file=sys.stderr)

        for m in models:
            if m['model'] not in all_models:
                all_models[m['model']] = (m['debit_b1'], m['bloc'])

    # Afficher en CSV
    print("modèle,débit_b1,bloc_source")
    for model_name in sorted(all_models.keys()):
        debit, bloc = all_models[model_name]
        print(f"{model_name},{debit},{bloc}.md")

    print(f"\nTotal : {len(all_models)} modèles extraits", file=sys.stderr)

if __name__ == "__main__":
    main()
