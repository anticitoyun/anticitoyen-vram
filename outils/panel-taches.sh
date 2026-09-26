#!/bin/bash
# Pièce 261b (poste2, ordre chef 26/09) : panel de tâches mini lm-eval contre un `acvram
# serve` DÉJÀ LANCÉ — sort le score par tâche, la moyenne et la pire tâche, pour une
# comparaison P1/P0 (critère Q19 : récupération moyenne ≥ 99 %, pire tâche jamais cachée
# derrière la moyenne).
#
# 261b (chef, après un 1er essai à quasi-hasard, 26/09) : `local-completions` en 5-shot
# BRUT (sans gabarit de conversation) donnait MMLU ≈ 20 % (le hasard, 4 choix) et GSM8K 0/50
# sur un modèle Instruct — l'INSTRUMENT était faux, pas le modèle. Corrigé : `local-chat-
# completions` + `--apply_chat_template` (le gabarit du modèle, via `/v1/chat/completions`)
# ET les variantes MMLU `*_generative` (`local-chat-completions` ne supporte PAS
# `loglikelihood` — `openai_completions.py:LocalChatCompletion.loglikelihood` lève
# `NotImplementedError`, vérifié dans le code).
#
# 2e correctif (même journée) : `*_generative` coupe la génération au 1er saut de ligne
# (`until: ["</s>", "\n"]`) et compare la 1re ligne au bit à la lettre attendue — conçu pour un
# modèle DE BASE qui répond "Answer: C" tout de suite, pas pour un modèle Instruct qui
# raisonne d'abord (vérifié sur échantillon réel : réponse coupée après « I need to find... »,
# jamais la lettre). Remplacé par `mmlu_flan_cot_zeroshot_*` : prompt "Let's think step by
# step", pas de coupe prématurée (stop sur `</s>`/`Q:`/`<|im_end|>` seulement), extraction par
# `The answer is X` (repli sur un motif `(X)` isolé) — conçu pour exactement ce cas.
#
# Sous-ensemble et graine FIXÉS ICI, jamais en argument : deux bras comparés (P0, P1, ou un
# bras et sa référence llama.cpp) doivent tourner sur EXACTEMENT le même tirage.
#   MMLU (generative, gabarit de conversation) : mmlu_high_school_mathematics_generative,
#   mmlu_professional_law_generative, mmlu_college_computer_science_generative — 100/tâche.
#   GSM8K (génération) : 50 questions.
#   graine : 1234 (python, numpy, torch, few-shot).
#
# `--log_samples` : conservé (jamais supprimé) sous `$SORTIE.echantillons/` — nécessaire à la
# vérification humaine de 5 échantillons (261b, demande chef) : réponse brute, extraite,
# attendue.
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
[ -x "$PY" ] || { echo "REFUS : $PY absent — installer d'abord (uv venv .venv-panel --python 3.12 && uv pip install --python .venv-panel 'lm-eval[api]' transformers)"; exit 66; }

GRAINE=1234
TACHES_MMLU="mmlu_flan_cot_zeroshot_high_school_mathematics,mmlu_flan_cot_zeroshot_professional_law,mmlu_flan_cot_zeroshot_college_computer_science"
LIMITE_MMLU=${LIMITE_MMLU:-100}
TACHE_GSM8K="gsm8k"
LIMITE_GSM8K=${LIMITE_GSM8K:-50}

D="${SORTIE}.echantillons"
# vide AVANT chaque prise : `lm_eval --output_path` AJOUTE un `results_*.json` horodaté sans
# jamais retirer les anciens — un `find | head -1` sur un dossier réutilisé (mêmes `--tasks`
# renommées entre deux essais, ex. 261b) reprenait un résultat PÉRIMÉ d'un essai précédent,
# silencieusement (aucune erreur, juste le mauvais fichier) : bogue trouvé en écrivant cette
# note, jamais publié.
rm -rf "$D"
mkdir -p "$D"

echo "=== MMLU generative (gabarit de conversation, limite $LIMITE_MMLU/tâche, graine $GRAINE)"
# max_gen_toks : defaut lm-eval 256, beaucoup trop court pour un CoT ("Let's think step by
# step") qui raisonne AVANT de conclure "The answer is X" -- verifie sur echantillon reel :
# reponses coupees en plein raisonnement, jamais la conclusion, filtered_resps = [invalid]
# partout, 0 % des DEUX cotes (acvram ET llama.cpp) -- signe du panel, pas du modele.
"$PY" -m lm_eval run --model local-chat-completions --apply_chat_template \
  --model_args "model=${SERVED},base_url=${BASE_URL}/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=${TOK},num_concurrent=1,max_retries=3" \
  --tasks "$TACHES_MMLU" --limit "$LIMITE_MMLU" --seed "$GRAINE" \
  --gen_kwargs "max_gen_toks=1536" \
  --output_path "$D/mmlu" --batch_size 1 --log_samples

echo "=== GSM8K (gabarit de conversation, limite $LIMITE_GSM8K, graine $GRAINE)"
"$PY" -m lm_eval run --model local-chat-completions --apply_chat_template \
  --model_args "model=${SERVED},base_url=${BASE_URL}/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=${TOK},num_concurrent=1,max_retries=3" \
  --tasks "$TACHE_GSM8K" --limit "$LIMITE_GSM8K" --seed "$GRAINE" \
  --output_path "$D/gsm8k" --batch_size 1 --log_samples

R_MMLU=$(find "$D/mmlu" -name "results_*.json" | head -1)
R_GSM8K=$(find "$D/gsm8k" -name "results_*.json" | head -1)
[ -n "$R_MMLU" ] || { echo "REFUS : aucun results_*.json pour MMLU"; exit 70; }
[ -n "$R_GSM8K" ] || { echo "REFUS : aucun results_*.json pour GSM8K"; exit 70; }

"$PY" "$ICI/outils/panel-taches-resume.py" "$R_MMLU" "$R_GSM8K" "$SORTIE"
echo "échantillons (log_samples) conservés sous $D"
