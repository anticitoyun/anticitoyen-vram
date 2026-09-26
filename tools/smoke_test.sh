#!/usr/bin/env bash
# Verification minimale de bout en bout sur une vraie machine : convertir un
# petit modele et le servir.
set -euo pipefail
MODEL="${1:?usage : smoke_test.sh /chemin/vers/modele-hf}"
OUT="${2:-/tmp/acvram-smoke}"

acvram doctor
acvram detect
acvram plan "$MODEL" --max-model-len 4096
acvram convert "$MODEL" -o "$OUT" --max-model-len 4096
acvram bench "$OUT" --what all

acvram serve "$OUT" --port 8123 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
sleep 20

curl -sf http://127.0.0.1:8123/v1/models | head -c 400; echo
curl -sf http://127.0.0.1:8123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"smoke","messages":[{"role":"user","content":"Dis bonjour en francais."}],"max_tokens":32}' \
  | head -c 600; echo
echo "test de fumee termine"
