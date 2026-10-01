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
# `.venv-panel` (refus sinon, comme `panel-taches.sh`) et `--je-sais-que-la-carte-est-libre`.
#
# NON EXERCÉ EN CONDITIONS RÉELLES À L'ÉCRITURE DE CETTE PIÈCE : `.venv-panel` n'existe pas sur
# ce poste (aucune prise, aucun lm-eval installé) — seul `--simule` est vérifié par les tests de
# cette pièce. `--executer` est écrit avec le même soin mais À VALIDER au premier essai réel
# (poste2/poste1, campagne-qualite-e50.py, pièce suivante).
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
PY_PANEL="$ICI/.venv-panel/bin/python"
[ -x "$PY_PANEL" ] || { echo "REFUS : $PY_PANEL absent — installer .venv-panel d'abord (voir outils/panel-taches.sh)" >&2; exit 66; }

echo "ÉCHEC : --executer n'est pas encore exercé sur ce poste (.venv-panel absent à l'écriture de cette pièce) — à compléter/valider à la première campagne réelle (poste2/poste1), voir le verdict e50.3 §7." >&2
exit 67
