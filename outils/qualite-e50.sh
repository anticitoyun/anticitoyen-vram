#!/bin/bash
# e50.3 § 7 (poste3, 01/10) — note un alias du menu sur la batterie E50 (méthode :
# acvram-memoire/revue/poste6-e50-3-methode-qualite-01-10.md, scellée AVANT toute mesure).
#
# `--simule` (DÉFAUT, REGLES §3 : aucune carte, aucun processus) : résout l'alias (dossier,
# tokenizer, étiquette thinking e50.1), imprime le plan exact (tâches, limites, max_gen_toks,
# « sans raisonnement » ou non) et sort 0 — rien n'est lancé.
# `--executer` : serveur acvram NEUF (`--no-prefix-cache`), lm-eval × 2 appels (MMLU+GSM8K en
# un seul, limite 30/30/30/40 ; HumanEval à part, génération seule, `--log_samples`, scoré par
# `outils/qualite-e50-humaneval-score.py` sous bac à sable, § 2 bis), S et étoile
# (`outils/qualite-e50-bareme.py`), écriture PAR `parc.py:ecrire_note` SEULEMENT (jamais une
# autre écriture du TSV des fiches), ligne ajoutée à `outils/qualite-e50.tsv`. Requiert
# un interprète lm-eval (refus sinon) et `--je-sais-que-la-carte-est-libre`. Interprète résolu
# dans cet ordre (chef, 01/10) : `$ACVRAM_LMEVAL_PY`, puis `$DEPOT/.venv-panel/bin/python`
# (emplacement réel attendu, `panel-taches.sh`), puis la copie figée temporaire de la campagne
# 275 — même règle que `tests/test_qualite_e50_taches_chargent.py`, pour que ce test vérifie le
# même lm-eval qu'une campagne réelle lancerait.
#
# e50.3 § 7 (poste4, 01/10) — `--executer` implémenté : harnais `serveur-bras.sh` (déjà
# éprouvé par tests/test_serveur_bras.py), 2 appels lm-eval (MMLU+GSM8K, puis HumanEval),
# agrégation (MMLU moyenne des 3 sous-tâches, GSM8K filtre flexible-extract), pas@1 HumanEval
# via le bac à sable, S composite, étoile, écriture TSV + `ecrire_note`.
# TESTÉ DE BOUT EN BOUT contre un FAUX serveur (`/v1/chat/completions` maison) ET un FAUX
# lm-eval (émule `python -m lm_eval run`, produit un `results_*.json`/`samples_*.jsonl` de la
# même forme) — valide la PLOMBERIE (lancement/arrêt serveur, agrégation, barème, écriture),
# jamais la qualité d'un vrai modèle ni le comportement réel du paquet lm-eval
# (`tests/test_qualite_e50_executer_faux_serveur.py`). AUCUN `.venv-panel` sous ce worktree
# (aucune prise, pas de carte) — à valider contre lm-eval et un `acvram-serveur` réels à la
# première campagne (poste2/poste1, `campagne-qualite-e50.py`).
#
# Usage : outils/qualite-e50.sh <alias> [--executer --je-sais-que-la-carte-est-libre]
set -euo pipefail
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ICI"
AL=${1:?usage: qualite-e50.sh <alias> [--executer --je-sais-que-la-carte-est-libre]}
shift || true
MODE=simule
CONFIRME=0
for a in "$@"; do
  case "$a" in
    --executer) MODE=executer ;;
    --simule) MODE=simule ;;
    --je-sais-que-la-carte-est-libre) CONFIRME=1 ;;
    *) echo "REFUS : argument inconnu « $a »" >&2; exit 64 ;;
  esac
done

PY="$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python"
# parc.py importe `gi` (PyGObject/GTK4) même pour lire une fiche — pas dans le venv de calcul
# (torch/triton) : /usr/bin/python3 (système) le porte, comme les scripts de parc/bin.
PY_GI=/usr/bin/python3
. "$ICI/outils/arbre-defaut.sh" 2>/dev/null || true   # bd jdp : présent sur main depuis 01/10 ; absent -> ACVRAM_ARBRE non posée, sans effet ici (pas d'import acvram)

# --- résolution de l'alias (sans construire de Modele/fenêtre : dossier, usage/refus/tps
# actuels, capacités e50.1 seulement). ---
PLAN_JSON=$("$PY_GI" - "$AL" <<'PYEOF'
import sys, json, os
sys.path.insert(0, "/usr/share/acvram-parc/lib")
sys.path.insert(0, "parc/lib")
from menu_modeles.parc import lire_tsv, deriver_capacites
from menu_modeles.config import NOTES, GGUF_TSV, VLLM_TSV, ACVRAM_TSV, VISION_TSV, CONFIG
import tomllib

alias = sys.argv[1]
conf = tomllib.load(CONFIG.open("rb")) if CONFIG.exists() else {"models": {}}
m = conf.get("models", {}).get(alias)
if m is None:
    print(json.dumps({"erreur": f"alias absent de config.toml : {alias}"})); sys.exit(0)
provider = m.get("provider")
notes = lire_tsv(NOTES, 5)
n = notes.get(alias) or [alias, "inconnu", "non mesuré", "non mesuré", "chat"]
gguf, vllm, acvr, vision = lire_tsv(GGUF_TSV, 3), lire_tsv(VLLM_TSV, 3), lire_tsv(ACVRAM_TSV, 3), lire_tsv(VISION_TSV, 2)
dossier = ""
if provider == "llamacpp" and alias in gguf:
    dossier = gguf[alias][0]
elif provider == "vllm" and alias in vllm:
    dossier = vllm[alias][0]
elif provider == "acvram" and alias in acvr:
    dossier = acvr[alias][0]
vs = (vision.get(alias) or [""])[0]
caps = deriver_capacites(alias, m.get("model", "?"), dossier, vs, m.get("capabilities", []))
print(json.dumps({
    "alias": alias, "provider": provider, "nom": m.get("model", "?"), "dossier": dossier,
    "refus": n[0], "tps": n[1], "usage": n[3], "thinking": "thinking" in caps,
}))
PYEOF
)
if python3 -c "import json,sys; sys.exit(0 if 'erreur' in json.loads('''$PLAN_JSON''') else 1)" 2>/dev/null; then
  echo "REFUS : $(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['erreur'])")" >&2
  exit 3
fi
DOSSIER=$(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['dossier'])")
THINKING=$(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['thinking'])")
REFUS=$(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['refus'])")
TPS=$(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['tps'])")
USAGE=$(python3 -c "import json; print(json.loads('''$PLAN_JSON''')['usage'])")
[ -n "$DOSSIER" ] || { echo "REFUS : pas de dossier connu pour $AL (TSV)" >&2; exit 3; }

SANS_RAISONNEMENT=non
[ "$THINKING" = "True" ] && SANS_RAISONNEMENT=oui   # e50.3 § 5 : détection automatique, jamais à la main

echo "=== qualite-e50 ($AL, mode=$MODE)"
echo "dossier   : $DOSSIER"
echo "thinking  : $THINKING (sans raisonnement si possible : $SANS_RAISONNEMENT)"
echo "tâches    : mmlu_e50_hsm(30) mmlu_e50_law(30) mmlu_e50_ccs(30) gsm8k_e50(40) humaneval_e50(40)"
echo "max_gen_toks : 768 (MMLU/GSM8K), 512 (HumanEval)"
echo "concurrence : 4 ; serveur neuf, --no-prefix-cache ; greedy (temperature=0, seed 1234)"

if [ "$MODE" = simule ]; then
  echo "=== --simule : rien lancé"
  exit 0
fi

[ "$CONFIRME" = 1 ] || { echo "REFUS : --executer exige --je-sais-que-la-carte-est-libre (REGLES §3)" >&2; exit 65; }
# Ordre de résolution (chef, 01/10, même règle que tests/test_qualite_e50_taches_chargent.py) :
# $ACVRAM_LMEVAL_PY, puis $ICI/.venv-panel (emplacement réel attendu, panel-taches.sh), puis la
# copie figée TEMPORAIRE de la campagne 275 — pour que le test de chargement vérifie le MÊME
# interpréteur que celui que cette campagne lancerait réellement.
PY_PANEL="${ACVRAM_LMEVAL_PY:-}"
if [ -z "$PY_PANEL" ] && [ -x "$ICI/.venv-panel/bin/python" ]; then
  PY_PANEL="$ICI/.venv-panel/bin/python"
fi
if [ -z "$PY_PANEL" ] && [ -x "$HOME/Bureau/Claude/travail/poste2-275-figee/.venv-panel/bin/python" ]; then
  PY_PANEL="$HOME/Bureau/Claude/travail/poste2-275-figee/.venv-panel/bin/python"
fi
[ -n "$PY_PANEL" ] && [ -x "$PY_PANEL" ] || {
  echo "REFUS : aucun interprète lm-eval trouvé (ACVRAM_LMEVAL_PY, $ICI/.venv-panel, ou la copie figée 275) — installer .venv-panel d'abord (voir outils/panel-taches.sh)" >&2
  exit 66
}
echo "lm-eval : $PY_PANEL"

# e50.3 § 7 (poste3, serveur+lm-eval, 01/10 ; poste4, 01/10) — harnais commun de lancement/arrêt
# déjà éprouvé contre un faux serveur (tests/test_serveur_bras.py) : jamais de `& $!` maison.
. "$ICI/outils/gpu/mesure/serveur-bras.sh"

PORT=${ACVRAM_E50_PORT:-8099}
NOM_SERVI=${ACVRAM_E50_SERVED_NAME:-$AL}
SORTIE=${ACVRAM_E50_SCRATCH:-$ICI/scratchpad/qualite-e50-$AL}   # surchargeable par les tests
rm -rf "$SORTIE"
mkdir -p "$SORTIE"

if [ -n "${ACVRAM_E50_LANCEUR:-}" ]; then
  # injection de test (faux serveur, tests/test_qualite_e50_executer_faux_serveur.py) : la
  # commande de lancement est fournie telle quelle, jamais acvram-serveur réel.
  # shellcheck disable=SC2086
  bras_servir "$PORT" "$SORTIE/serveur.log" $ACVRAM_E50_LANCEUR "$PORT" "$NOM_SERVI" \
    || { echo "REFUS : lancement du serveur (injecté) échoué" >&2; exit 70; }
else
  bras_servir "$PORT" "$SORTIE/serveur.log" acvram-serveur "$DOSSIER" 4096 --no-prefix-cache --port "$PORT" \
    || { echo "REFUS : lancement du serveur échoué" >&2; exit 70; }
fi
bras_pret "$PORT" "$NOM_SERVI" "$BRAS_PID" "${ACVRAM_E50_DELAI_S:-480}" \
  || { bras_arreter "$BRAS_PID" "$PORT" 2>/dev/null || true; exit 71; }

BASE_URL="http://127.0.0.1:$PORT"
TOK=${ACVRAM_E50_TOKENIZER:-$DOSSIER}

echo "=== MMLU(90)+GSM8K(40), num_concurrent=4, seed 1234"
"$PY_PANEL" -m lm_eval run --model local-chat-completions --apply_chat_template \
  --include_path "$ICI/outils/lm_eval_taches" \
  --model_args "model=${NOM_SERVI},base_url=${BASE_URL}/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=${TOK},num_concurrent=4,max_retries=3" \
  --tasks "mmlu_e50_hsm,mmlu_e50_law,mmlu_e50_ccs,gsm8k_e50" --seed 1234 \
  --output_path "$SORTIE/mmlu_gsm8k" --batch_size 1 --log_samples \
  || { echo "ÉCHEC : lm-eval MMLU+GSM8K" >&2; bras_arreter "$BRAS_PID" "$PORT" 2>/dev/null || true; exit 72; }

echo "=== HumanEval(40), génération seule (pas@1 à part, § 2 bis)"
"$PY_PANEL" -m lm_eval run --model local-chat-completions --apply_chat_template \
  --include_path "$ICI/outils/lm_eval_taches" \
  --model_args "model=${NOM_SERVI},base_url=${BASE_URL}/v1/chat/completions,tokenizer_backend=huggingface,tokenizer=${TOK},num_concurrent=4,max_retries=3" \
  --tasks "humaneval_e50" --seed 1234 \
  --output_path "$SORTIE/humaneval" --batch_size 1 --log_samples \
  || { echo "ÉCHEC : lm-eval HumanEval" >&2; bras_arreter "$BRAS_PID" "$PORT" 2>/dev/null || true; exit 73; }

bras_arreter "$BRAS_PID" "$PORT" || echo "AVERTISSEMENT : arrêt du serveur incertain" >&2

R_MG=$(find "$SORTIE/mmlu_gsm8k" -name "results_*.json" | head -1)
SAMPLES_HE=$(find "$SORTIE/humaneval" -name "samples_humaneval_e50_*.jsonl" | head -1)
[ -n "$R_MG" ] || { echo "REFUS : aucun results_*.json pour MMLU+GSM8K" >&2; exit 74; }
[ -n "$SAMPLES_HE" ] || { echo "REFUS : aucun samples_humaneval_e50_*.jsonl" >&2; exit 74; }

# Agrégation MMLU (3 sous-tâches, moyenne simple : 30 items chacune) + GSM8K flexible-extract
# (méthode § 2 : le filtre strict punit le format, pas le calcul — on garde flexible).
read -r ACC_MMLU ACC_GSM8K <<EOF
$(python3 - "$R_MG" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
r = d["results"]
mmlu = [r[t]["exact_match,get-answer-v275"] for t in ("mmlu_e50_hsm", "mmlu_e50_law", "mmlu_e50_ccs")]
print(sum(mmlu) / len(mmlu), r["gsm8k_e50"]["exact_match,flexible-extract"])
PYEOF
)
EOF

HE_JSON=$("$PY_GI" "$ICI/outils/qualite-e50-humaneval-score.py" "$SAMPLES_HE")
PASS_HE=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['pass_at_1'])" "$HE_JSON")
S_COMPOSITE=$(python3 -c "print(($ACC_MMLU+$ACC_GSM8K+$PASS_HE)/3)")
ETOILE=$("$PY" "$ICI/outils/qualite-e50-bareme.py" "$S_COMPOSITE" "$ACC_MMLU" "$ACC_GSM8K" "$PASS_HE")

echo "S=$S_COMPOSITE mmlu=$ACC_MMLU gsm8k=$ACC_GSM8K humaneval=$PASS_HE -> $ETOILE"

HEAD=$(git -C "$ICI" rev-parse --short HEAD)
DATE_J=$(date +%d/%m)
TSV_E50=${ACVRAM_E50_TSV:-$ICI/outils/qualite-e50.tsv}   # surchargeable par les tests (faux serveur)
# python3, pas `printf %f` : LC_NUMERIC peut être fr_FR (virgule décimale), printf refuserait
# alors un nombre à point — trouvé en écrivant cette pièce (test faux serveur, locale du poste).
LC_NUMERIC=C python3 -c "
import sys
al, head, date_j, etoile, s, mmlu, gsm8k, he, sans = sys.argv[1:10]
print(f'{al}\t{head}\t{date_j}\t{etoile}\t{float(s):.4f}\t{float(mmlu):.4f}\t{float(gsm8k):.4f}\t{float(he):.4f}\t{sans}')
" "$AL" "$HEAD" "$DATE_J" "$ETOILE" "$S_COMPOSITE" "$ACC_MMLU" "$ACC_GSM8K" "$PASS_HE" "$SANS_RAISONNEMENT" \
  >> "$TSV_E50"

SUFFIXE=""
[ "$SANS_RAISONNEMENT" = oui ] && SUFFIXE=" sans raisonnement"
QUAL_TXT=$(python3 -c "print(f'$ETOILE S {$S_COMPOSITE:.2f}$SUFFIXE ($DATE_J, $HEAD)')")

"$PY_GI" - "$AL" "$REFUS" "$TPS" "$QUAL_TXT" "$USAGE" <<'PYEOF'
import sys
sys.path.insert(0, "/usr/share/acvram-parc/lib")
sys.path.insert(0, "parc/lib")
from menu_modeles.parc import ecrire_note
alias, refus, tps, qual, usage = sys.argv[1:6]
ecrire_note(alias, refus, tps, qual, usage)
PYEOF
echo "écrit : $TSV_E50, note menu ($QUAL_TXT)"
