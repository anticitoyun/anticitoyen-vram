#!/usr/bin/env bash
# kv31b levier 1 — prise sur carte (≤ 15 min) : chauffe d'un dense à 32 768 avec la réserve de préfill plafonnée.
# Scellé (prédiction S1-S6) : acvram-memoire/revue/poste6-reserve-plafonnee-scelle-carte-30-09.md — écrit AVANT.
# Moteur depuis l'ARBRE (PYTHONPATH), pas le paquet (0.7.16 sans le levier 1) ; sous carte.sh ; relevés début/fin ;
# journal lu par ses lignes [acvram] seulement (§ 6). Usage : prise-reserve-plafonnee-kv31b.sh [alias] [ctx]
set -uo pipefail
ICI=$(dirname "$(readlink -f "$0")"); DEPOT=$(cd "$ICI/../../.." && pwd)
ALIAS="${1:-acvram-gemma-4-31b-it-nvfp4-4sur6-vision-nvfp4}"; CTX="${2:-32768}"
COMMIT_ATTENDU="${COMMIT_ATTENDU:-}"; PORT="${PORT:-8093}"
PY="${PY:-$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python}"
SORTIE="${SORTIE:-$DEPOT/scratchpad/poste6-kv31b-carte-$(date +%Y%m%d-%H%M)}"; mkdir -p "$SORTIE"
export ACVRAM_POSTE="${ACVRAM_POSTE:-poste6}" ACVRAM_DUREE_MAX="${ACVRAM_DUREE_MAX:-900}" PYTHONPATH="$DEPOT" ACVRAM_ARBRE="${ACVRAM_ARBRE:-$DEPOT}"
head=$(git -C "$DEPOT" rev-parse --short=9 HEAD)
if [ -n "$COMMIT_ATTENDU" ] && [[ "$head" != "$COMMIT_ATTENDU"* ]]; then echo "ÉCHEC : HEAD $head ≠ $COMMIT_ATTENDU"; exit 2; fi
dossier=$(grep "^$ALIAS	" "$HOME/TSV/acvram-chemins.tsv" | cut -f2); [ -d "$dossier" ] || { echo "ÉCHEC : alias $ALIAS inconnu"; exit 3; }
releve() { { date +%FT%T; uptime; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader;
             nvidia-smi --query-gpu=memory.used,memory.free,power.limit,clocks.sm --format=csv,noheader; } >> "$SORTIE/releve-$1.txt" 2>&1; }
echo "# HEAD $head · poste $ACVRAM_POSTE · durée max ${ACVRAM_DUREE_MAX}s · alias $ALIAS · ctx $CTX · port $PORT · sortie $SORTIE"
releve debut
LOG="$SORTIE/serveur.log"; t0=$SECONDS
srv=$(ACVRAM_TYPE=service ACVRAM_NOM="$ALIAS" ACVRAM_SERVICE_LOG="$LOG" ACVRAM_CARTE=0 \
      "$DEPOT/outils/carte.sh" "$PY" -m acvram.cli serve "$dossier" --port "$PORT" --served-name "$ALIAS" --max-model-len "$CTX" --speculative ngram) \
  || { echo "ÉCHEC : carte.sh a refusé le verrou"; releve fin; exit 4; }
echo "# serveur PID $srv, journal $LOG"
rc=1
while [ $((SECONDS - t0)) -lt 840 ]; do
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then rc=0; break; fi
  kill -0 "$srv" 2>/dev/null || { echo "# serveur mort au démarrage"; break; }
  sleep 3
done
echo "# prêt=$rc après $((SECONDS - t0)) s"
echo "# --- lignes [acvram] décisives (plafond, exil, KV, chauffe, refus) :"
grep -E '^\[acvram\] (MLP dense par tranches|plan réajusté|budget KV|fenêtre qui tient|régime|ATTENTION — exil)|RuntimeError|refus :|OutOfMemory' "$LOG" | cut -c1-260 | tail -14
if [ "$rc" = 0 ]; then
  code=$(curl -s -m 120 -o "$SORTIE/completion.json" -w '%{http_code}' -H 'Content-Type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"prompt\":\"Bonjour, je\",\"max_tokens\":8,\"temperature\":0}" "http://127.0.0.1:$PORT/v1/completions")
  echo "# complétion 8 jetons : HTTP $code ($(wc -c < "$SORTIE/completion.json") o — contenu NON lu, § 6)"
fi
kill "$srv" 2>/dev/null; for _ in $(seq 1 40); do kill -0 "$srv" 2>/dev/null || break; sleep 1; done; kill -9 "$srv" 2>/dev/null || true
sleep 3; releve fin
echo "# fin : $((SECONDS - t0)) s ; relevés $SORTIE/releve-{debut,fin}.txt ; rc $rc"
exit "$rc"
