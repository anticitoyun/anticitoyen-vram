#!/bin/bash
# Pièce 275 (poste2, ordre chef, 26/09) : garde de régression de QUALITÉ, une commande unique
# pour toute pièce qui change la sortie servie — rend TENU ou FAUX contre une référence figée.
#
# Référence : générée UNE FOIS par modèle sur le bras par défaut (0.7.3), hors git, sous
# ~/.cache/acvram/qualite-275/<alias>/ (panel par tâche + PPL). `<bras>` est soit `defaut`
# (aucune variable posée, code tel quel), soit `VAR=valeur[,VAR2=valeur2...]` — les variables
# d'environnement à poser AVANT `acvram serve`/`acvram eval` pour l'essai (ex.
# `ACVRAM_MARLIN_PAR_LIGNE=1`).
#
# Règle de décision (scellée AVANT toute référence, `revue/poste2-piece275-scelle-26-09.md`) :
#   TENU  si McNemar (p > 0,05) TENU sur les 4 tâches ET PPL du bras dans la référence ± 1 %.
#   FAUX  sinon (au moins une tâche McNemar p <= 0,05, OU PPL hors ± 1 %).
#
# Usage : ATTENDU=<commit> outils/qualite.sh <alias> <defaut|VAR=val[,VAR2=val2]>
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ATTENDU:?ATTENDU=<commit court de HEAD> requis (chaque sous-prise le revérifie)}"
AL=$1; BRAS=$2
REF=$HOME/.cache/acvram/qualite-275/$AL
[ -d "$REF" ] || { echo "REFUS : pas de référence pour $AL sous $REF — générer d'abord (scratchpad/poste2-p275-26-09/generer-reference-v2.sh)"; exit 66; }

SEUIL_PPL=0.01  # ± 1 %, REGLES (convention PPL du dépôt)
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

# --- variables du bras, exportées pour les sous-appels ---
if [ "$BRAS" != defaut ]; then
  IFS=',' read -ra PAIRES <<< "$BRAS"
  for p in "${PAIRES[@]}"; do
    export "${p?}"
  done
fi

echo "=== PPL ($AL, bras=$BRAS)"
CORPUS=/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt
PY=$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python
M=$($PY -c "import sys; sys.path.insert(0,'outils'); from racine_modeles import racine_modeles; print(racine_modeles())")/$AL
ACVRAM_NOM="poste2-qualite-${AL}-${BRAS//[,=]/_}-ppl" ATTENDU="${ATTENDU:?commit}" outils/carte.sh \
  bash -c "CUDA_VISIBLE_DEVICES=0 $PY -m acvram.cli eval '$M' --corpus '$CORPUS' --window 2048 --stride 2048 --min-context 256 --json > '$D/ppl.json'"

PPL_BRAS=$(python3 -c "
import json, re
txt = open('$D/ppl.json').read()
d = json.loads(txt[txt.rfind(chr(10)+'[')+1:])
print(d[0]['perplexity'])
")
PPL_REF=$(python3 -c "
import json, re
txt = open('$REF/ppl.json').read()
d = json.loads(txt[txt.rfind(chr(10)+'[')+1:])
print(d[0]['perplexity'])
")
RATIO_PPL=$(python3 -c "print(abs($PPL_BRAS/$PPL_REF - 1))")
PPL_TENU=$(python3 -c "print('1' if abs($PPL_BRAS/$PPL_REF - 1) <= $SEUIL_PPL else '0')")
echo "PPL bras=$PPL_BRAS reference=$PPL_REF ecart_relatif=$RATIO_PPL seuil=$SEUIL_PPL tenu=$PPL_TENU"

declare -A LIMITES=(
  [gsm8k]=250
  [mmlu_flan_cot_fewshot_high_school_mathematics]=150
  [mmlu_flan_cot_fewshot_professional_law]=150
  [mmlu_flan_cot_fewshot_college_computer_science]=150
)
declare -A FILTRES=(
  [gsm8k]=strict-match
  [mmlu_flan_cot_fewshot_high_school_mathematics]=get-answer
  [mmlu_flan_cot_fewshot_professional_law]=get-answer
  [mmlu_flan_cot_fewshot_college_computer_science]=get-answer
)

MCNEMAR_TENU=1
for tache in "${!LIMITES[@]}"; do
  n=${LIMITES[$tache]}
  sortie="$D/${tache}.json"
  echo "=== $tache ($AL, bras=$BRAS, n=$n) $(date +%T)"
  ACVRAM_NOM="poste2-qualite-${AL}-${BRAS//[,=]/_}-${tache}" ATTENDU="${ATTENDU:?commit}" outils/carte.sh \
    bash scratchpad/poste2-p275-26-09/prise-tache-275.sh "$AL" "$tache" "$n" "$sortie" > "$D/${tache}.log" 2>&1

  ECH_BRAS=$(find "$sortie.echantillons" -name "samples_${tache}_*.jsonl" | head -1)
  ECH_REF=$(find "$REF/${tache}.json.echantillons" -name "samples_${tache}_*.jsonl" | head -1)
  [ -n "$ECH_BRAS" ] || { echo "REFUS : pas d'echantillons pour $tache (bras)"; exit 70; }
  [ -n "$ECH_REF" ] || { echo "REFUS : pas d'echantillons pour $tache (reference)"; exit 70; }

  R=$(python3 scratchpad/poste2-p237d-26-09/mcnemar-ic-237d.py "$ECH_REF" "$ECH_BRAS" "${FILTRES[$tache]}" 2000)
  echo "$R"
  P=$(echo "$R" | python3 -c "import json,sys; print(json.loads(sys.stdin.read().split('RESULTAT_237D ',1)[1])['mcnemar_p'])")
  TENU=$(python3 -c "print('1' if $P > 0.05 else '0')")
  echo "$tache : mcnemar_p=$P tenu=$TENU"
  [ "$TENU" = "1" ] || MCNEMAR_TENU=0
done

echo "=== VERDICT ($AL, bras=$BRAS)"
echo "McNemar (4 taches) tenu=$MCNEMAR_TENU ; PPL tenu=$PPL_TENU (ecart $RATIO_PPL, seuil $SEUIL_PPL)"
if [ "$MCNEMAR_TENU" = "1" ] && [ "$PPL_TENU" = "1" ]; then
  echo "TENU"
  exit 0
else
  echo "FAUX"
  exit 1
fi
