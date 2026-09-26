#!/bin/bash
# Pièce 261 (poste2, ordre chef 26/09, reprise de poste4 partie sur la 070b) : panel de
# tâches mini lm-eval contre un `acvram serve` DÉJÀ LANCÉ (ce script ne lance aucun serveur) —
# sort le score par tâche, la moyenne et la pire tâche, pour une comparaison P1/P0 (critère
# Q19 : récupération moyenne ≥ 99 %, pire tâche jamais cachée derrière la moyenne).
#
# Sous-ensemble et graine FIXÉS ICI, jamais en argument : deux bras comparés (P0, P1) doivent
# tourner sur EXACTEMENT le même tirage, sinon la comparaison ne vaut rien.
#   MMLU (loglikelihood, echo+logprobs de /v1/completions) : mmlu_high_school_mathematics,
#   mmlu_professional_law, mmlu_college_computer_science — 100 questions chacune.
#   GSM8K (génération) : 50 questions.
#   graine : 1234 (python, numpy, torch, few-shot).
#
# Usage : panel-taches.sh <served_name> <base_url> <repertoire_tokenizer> <sortie.json>
#   ex.  : panel-taches.sh coder http://127.0.0.1:8151 \
#            /mnt/AI_GENERATOR/models_acvram/Qwen3-Coder-30B-A3B-nvfp4 sortie.json
#
# lm-eval vit dans .venv-panel, À LA RACINE DE CE WORKTREE SEULEMENT (jamais le .venv du
# dépôt : dépendances d'évaluation séparées de torch/triton du serveur, jamais installées
# dedans).
set -euo pipefail
SERVED=$1; BASE_URL=${2%/}; TOK=$3; SORTIE=$4
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ICI/.venv-panel/bin/python"
[ -x "$PY" ] || { echo "REFUS : $PY absent — installer d'abord (uv venv .venv-panel --python 3.12 && uv pip install --python .venv-panel lm-eval)"; exit 66; }

GRAINE=1234
TACHES_MMLU="mmlu_high_school_mathematics,mmlu_professional_law,mmlu_college_computer_science"
LIMITE_MMLU=100
TACHE_GSM8K="gsm8k"
LIMITE_GSM8K=50

D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

echo "=== MMLU (loglikelihood, limite $LIMITE_MMLU/tâche, graine $GRAINE)"
"$PY" -m lm_eval run --model local-completions \
  --model_args "model=${SERVED},base_url=${BASE_URL}/v1/completions,tokenizer=${TOK},tokenizer_backend=huggingface,num_concurrent=1,max_retries=3" \
  --tasks "$TACHES_MMLU" --limit "$LIMITE_MMLU" --seed "$GRAINE" \
  --output_path "$D/mmlu" --batch_size 1

echo "=== GSM8K (génération, limite $LIMITE_GSM8K, graine $GRAINE)"
"$PY" -m lm_eval run --model local-completions \
  --model_args "model=${SERVED},base_url=${BASE_URL}/v1/completions,tokenizer=${TOK},tokenizer_backend=huggingface,num_concurrent=1,max_retries=3" \
  --tasks "$TACHE_GSM8K" --limit "$LIMITE_GSM8K" --seed "$GRAINE" \
  --output_path "$D/gsm8k" --batch_size 1

R_MMLU=$(find "$D/mmlu" -name "results_*.json" | head -1)
R_GSM8K=$(find "$D/gsm8k" -name "results_*.json" | head -1)
[ -n "$R_MMLU" ] || { echo "REFUS : aucun results_*.json pour MMLU"; exit 70; }
[ -n "$R_GSM8K" ] || { echo "REFUS : aucun results_*.json pour GSM8K"; exit 70; }

"$PY" "$ICI/outils/panel-taches-resume.py" "$R_MMLU" "$R_GSM8K" "$SORTIE"
