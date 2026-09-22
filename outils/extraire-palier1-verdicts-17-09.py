#!/usr/bin/env python3
"""
Extrait les PPL et débits depuis les 8 blocs verdict palier 1.
Génère un CSV de modèles → (PPL, juge, débit, verdict_file).

Usage :
  python extraire-palier1-verdicts-17-09.py > palier1-mapping-17-09.csv
"""

import re
import sys
from pathlib import Path

REVUE = Path(__file__).parent.parent / "acvram-memoire/revue"

def parse_bloc_verdict(bloc_path):
    """
    Parse un fichier verdict-palier1-bloc*.md.
    Retourne liste de (nom_modèle, ppl_géo, ppl_med, débit_b1, régime, verdict_file).

    Format attendu dans le verdict :
    - "Modèle-X-nvfp4 : b=1 **123,4 t/s** · PPL **0,987 géo** (med 0,993)"
    - "Modèle-Y-gguf : b=1 **refusé** (budget KV) ; PPL absolues **non classables**"
    """
    results = []
    verdict_name = bloc_path.stem  # ex: "verdict-palier1-bloc0-17-09"

    with open(bloc_path) as f:
        content = f.read()

    # Patterns génériques pour les modèles et leurs métriques
    # Pattern 1 : PPL avec juge (géo ou médiane)
    ppl_pattern = r'PPL\s+\*\*([0-9,]+)\s+(géo|médiane)\*\*(?:\s+\((?:med|médiane)\s+([0-9,]+)\))?'

    # Pattern 2 : débit b=1
    debit_pattern = r'b=1\s+\*\*([0-9,]+\s+t/s|\d+\.?\d+|refusé)'

    # Matcher les paragraphes par modèle (ligne commençant par backtick ou nom)
    model_blocks = re.split(r'\n(?=[A-Z][A-Za-z0-9\-\.]+[\s:]*)', content)

    for block in model_blocks:
        lines = block.strip().split('\n')
        if not lines or not lines[0].strip():
            continue

        # Extraire le nom du modèle (première ligne, souvent entre backticks)
        first_line = lines[0]
        model_name = first_line.strip()
        if model_name.startswith('`'):
            model_name = model_name.strip('`')
        model_name = re.sub(r'\s.*', '', model_name)  # Juste le nom

        if not model_name or len(model_name) < 3:
            continue

        # Chercher les métriques dans ce bloc
        block_text = '\n'.join(lines)

        ppl_match = re.search(ppl_pattern, block_text)
        ppl_val = ppl_med = juge = None
        if ppl_match:
            ppl_val = ppl_match.group(1)
            juge = ppl_match.group(2)
            ppl_med = ppl_match.group(3)

        debit_match = re.search(debit_pattern, block_text)
        debit_b1 = debit_match.group(1) if debit_match else None

        # Détecter le régime (format)
        regime = "inconnu"
        if "nvfp4" in block_text:
            regime = "nvfp4"
        elif "GGUF" in block_text or "gguf" in block_text:
            regime = "GGUF"
        elif "EXL3" in block_text or "exl3" in block_text:
            regime = "EXL3"
        elif "vLLM" in block_text or "vllm" in block_text:
            regime = "vLLM"

        # Créer la ligne de résultat
        if ppl_val:  # Seulement si une PPL a été trouvée
            results.append({
                'model': model_name,
                'ppl': ppl_val,
                'juge': juge,
                'ppl_med': ppl_med,
                'debit_b1': debit_b1,
                'regime': regime,
                'verdict_file': verdict_name,
            })

    return results

def main():
    # Lire tous les blocs
    all_models = []
    for bloc_num in range(8):
        bloc_path = REVUE / f"verdict-palier1-bloc{bloc_num}-17-09.md"
        if not bloc_path.exists():
            print(f"Attention : {bloc_path.name} introuvable", file=sys.stderr)
            continue

        models = parse_bloc_verdict(bloc_path)
        all_models.extend(models)
        print(f"Bloc {bloc_num} : {len(models)} modèles extraits", file=sys.stderr)

    # Afficher en CSV
    print("modèle,PPL,juge,PPL_médiane,débit_b1,régime,fichier_verdict")
    for m in all_models:
        ppl_med = m['ppl_med'] or ''
        debit = m['debit_b1'] or ''
        print(
            f"{m['model']},{m['ppl']},{m['juge']},{ppl_med},{debit},{m['regime']},{m['verdict_file']}.md"
        )

    print(f"\nTotal : {len(all_models)} modèles extraits", file=sys.stderr)

if __name__ == "__main__":
    main()
