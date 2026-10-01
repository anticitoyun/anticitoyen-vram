#!/usr/bin/env bash
# kv31b levier 2 étape 1 — preuve carte S1 (≤ 15 min par bras, reprenable) : préfill par morceaux au bit du seul tenant.
# Scellé (P1-P6, bras A1/A2/B) : acvram-memoire/revue/poste6-s1-morceaux-scelle-carte-30-09.md — écrit AVANT.
# Un processus par bras (ACVRAM_PREFILL_MORCEAU lu à l'import) ; moteur depuis l'ARBRE (PYTHONPATH) ; sous carte.sh ; relevés
# début/fin ; journal lu par ses lignes [acvram] seulement ; le texte généré n'est jamais affiché (§ 6). Comparaison à sec :
# s1-morceaux-comparer.py. Usage : prise-s1-morceaux-kv31b.sh A1|A2|B ; env : ALIAS, CTX, SORTIE, OPTIONS_SERVE (ex. --no-prefix-cache)
set -uo pipefail
ICI=$(dirname "$(readlink -f "$0")"); DEPOT=$(cd "$ICI/../../.." && pwd)
BRAS="${1:?bras A1, A2 ou B}"
case "$BRAS" in A1|A2) MORCEAU=0 ;; B) MORCEAU=4096 ;; *) echo "ÉCHEC : bras $BRAS inconnu (A1, A2, B)"; exit 2 ;; esac
ALIAS="${ALIAS:-acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4}"; CTX="${CTX:-10240}"
INVITE_COMMIT="${INVITE_COMMIT:-87d8bfe0a}"; INVITE_SHA="${INVITE_SHA:-0640377149ab8d46803ffc7e98e21e4036b5c4e9f96dad7793c9eb5a05fea1c5}"
COMMIT_ATTENDU="${COMMIT_ATTENDU:-}"; PORT="${PORT:-8093}"
PY="${PY:-$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python}"
SORTIE="${SORTIE:-$DEPOT/scratchpad/poste6-s1-$BRAS}"; mkdir -p "$SORTIE"
export ACVRAM_POSTE="${ACVRAM_POSTE:-poste6}" ACVRAM_DUREE_MAX="${ACVRAM_DUREE_MAX:-900}" PYTHONPATH="$DEPOT"
export ACVRAM_PREFILL_MORCEAU="$MORCEAU" ACVRAM_SAMPLER_LENT=1
head=$(git -C "$DEPOT" rev-parse --short=9 HEAD)
if [ -n "$COMMIT_ATTENDU" ] && [[ "$head" != "$COMMIT_ATTENDU"* ]]; then echo "ÉCHEC : HEAD $head ≠ $COMMIT_ATTENDU"; exit 2; fi
dossier=$(grep "^$ALIAS	" "$HOME/TSV/acvram-chemins.tsv" | cut -f2); [ -d "$dossier" ] || { echo "ÉCHEC : alias $ALIAS inconnu"; exit 3; }
# Invite figée par son objet git, jamais l'arbre : même texte pour les trois bras, quel que soit HEAD.
git -C "$DEPOT" show "$INVITE_COMMIT:README.md" > "$SORTIE/invite.txt" || { echo "ÉCHEC : invite $INVITE_COMMIT:README.md illisible"; exit 3; }
sha=$(sha256sum "$SORTIE/invite.txt" | cut -d' ' -f1)
[ "$sha" = "$INVITE_SHA" ] || { echo "ÉCHEC : invite sha256 $sha ≠ scellé"; exit 3; }
"$PY" - "$SORTIE/invite.txt" "$ALIAS" > "$SORTIE/requete.json" <<'EOF'
import json, sys
print(json.dumps({"model": sys.argv[2], "prompt": open(sys.argv[1], encoding="utf-8").read(),
                  "max_tokens": 32, "temperature": 0, "logprobs": 10}, ensure_ascii=False))
EOF
releve() { { date +%FT%T; uptime; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader;
             nvidia-smi --query-gpu=memory.used,memory.free,power.limit,clocks.sm --format=csv,noheader; } >> "$SORTIE/releve-$1.txt" 2>&1; }
echo "# bras $BRAS · morceau $MORCEAU · HEAD $head · poste $ACVRAM_POSTE · durée max ${ACVRAM_DUREE_MAX}s · ctx $CTX · invite $INVITE_COMMIT:README.md $(wc -c < "$SORTIE/invite.txt") o · sortie $SORTIE"
releve debut
LOG="$SORTIE/serveur.log"; t0=$SECONDS
srv=$(ACVRAM_TYPE=service ACVRAM_NOM="s1-$BRAS-$ALIAS" ACVRAM_SERVICE_LOG="$LOG" ACVRAM_CARTE=0 \
      "$DEPOT/outils/carte.sh" "$PY" -m acvram.cli serve "$dossier" --port "$PORT" --served-name "$ALIAS" --max-model-len "$CTX" \
      --speculative none --max-batch 1 ${OPTIONS_SERVE:-}) \
  || { echo "ÉCHEC : carte.sh a refusé le verrou"; releve fin; exit 4; }
echo "# serveur PID $srv, journal $LOG"
rc=1
while [ $((SECONDS - t0)) -lt 780 ]; do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then rc=0; break; fi
  kill -0 "$srv" 2>/dev/null || { echo "# serveur mort au démarrage"; break; }
  sleep 3
done
echo "# prêt=$rc après $((SECONDS - t0)) s"
if [ "$rc" = 0 ]; then
  code=$(curl -s -m 300 -o "$SORTIE/completion.json" -w '%{http_code}' -H 'Content-Type: application/json' \
    --data-binary "@$SORTIE/requete.json" "http://127.0.0.1:$PORT/v1/completions")
  curl -s -m 10 -o "$SORTIE/metrics.json" "http://127.0.0.1:$PORT/metrics"
  echo "# complétion : HTTP $code ($(wc -c < "$SORTIE/completion.json") o — contenu NON lu, § 6) ; metrics $(wc -c < "$SORTIE/metrics.json") o"
  [ "$code" = 200 ] || rc=5
fi
echo "# --- lignes [acvram] décisives (régime, morceaux, exil, KV, fenêtre, refus) :"
grep -E '^\[acvram\] (régime|morceaux@|ACVRAM_PREFILL_MORCEAU|plan réajusté|budget KV|fenêtre qui tient|MLP dense par tranches)|RuntimeError|refus :|OutOfMemory' "$LOG" | cut -c1-260 | tail -14
kill "$srv" 2>/dev/null; for _ in $(seq 1 40); do kill -0 "$srv" 2>/dev/null || break; sleep 1; done; kill -9 "$srv" 2>/dev/null || true
sleep 3; releve fin
echo "# fin : $((SECONDS - t0)) s ; relevés $SORTIE/releve-{debut,fin}.txt ; rc $rc"
exit "$rc"
