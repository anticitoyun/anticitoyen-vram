#!/bin/bash
# Pièce 275 (poste2, ordre chef, 30/09) : vérifie qu'une référence qualite-275 est au
# format COURANT (échantillons nommés d'après la tâche lm-eval actuelle, mmlu_v275_*) et
# REFUSE NOMMÉMENT sinon — trouvé le 30/09 : une référence générée avant le renommage des
# tâches par la pièce 275d (26/09) garde ses fichiers d'échantillons sous l'ANCIEN nom
# (mmlu_flan_cot_fewshot_*) même après une « re-notation sans nouvelle génération » qui ne
# met à jour que panel.json, jamais les fichiers sur disque — qualite.sh plantait alors sur
# un `find` sans message clair, APRÈS avoir dépensé une prise carte.sh pour rien.
# Usage : verifier-reference-275.sh <ref_dossier>
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 1 ] || { echo "Usage : verifier-reference-275.sh <ref_dossier>" >&2; exit 2; }
AL=$1
REF=$HOME/.cache/acvram/qualite-275/$AL
TACHES=(gsm8k mmlu_v275_high_school_mathematics mmlu_v275_professional_law mmlu_v275_college_computer_science)

[ -f "$REF/ppl.json" ] || {
  echo "REFUS : pas de référence pour $AL sous $REF — générer d'abord (scratchpad/poste2-p275-26-09/generer-reference-v2.sh)"
  exit 66
}

for t in "${TACHES[@]}"; do
  ECH="$REF/${t}.json.echantillons"
  if ! find "$ECH" -name "samples_${t}_*.jsonl" 2>/dev/null | grep -q .; then
    echo "REFUS : référence $AL au format ANCIEN — aucun échantillon '$t' sous $ECH" \
         "(généré avant le renommage des tâches, pièce 275d, 26/09 ; la re-notation ne met" \
         "à jour que panel.json, jamais les fichiers) — régénérer avec generer-reference-v2.sh"
    exit 72
  fi
done

echo "OK : référence $AL au format courant"
