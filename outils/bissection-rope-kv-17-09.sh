#!/bin/bash
# Bissection du noyau rope_kv en situ (poste3 e31fa6f : ROPE_KV=1 corrompt le
# cache pas à pas — PPL décodage 1 722 pour 5,54 — alors que d462c24 rendait
# 0,9959 et que les tests unitaires sont verts). Le noyau est un seul fichier :
# on essaie chaque version entre la saine et la courante avec le test cassant
# de 27 s, sans rien changer d'autre dans l'arbre. À lancer sous le verrou,
# depuis la racine du dépôt ; restaure le fichier à la fin.
#
#   outils/bissection-rope-kv-17-09.sh   (≈ 3 min)
#
# Prédiction (poste4) : sain jusqu'à df4db39, cassé à partir de 0ab28f3
# (inverses en fp64) ou de acd9426 (libdevice.rint) — la seule ligne à
# retrouver, le reste du noyau n'a pas changé.
set -u
F=acvram/kernels/rope_kv.py
SAUVE=$(mktemp); cp $F "$SAUVE"; trap 'cp "$SAUVE" '"$F"'; rm -f "$SAUVE"' EXIT
for c in d462c24 df4db39 0ab28f3 8b85d2d acd9426 ece3bed; do
    git show "$c:$F" > $F || { echo "$c : fichier absent"; continue; }
    printf '%s  ' "$c"
    ACVRAM_ROPE_KV=1 ACVRAM_KV_FORMAT=int8 outils/carte.sh .venv/bin/python scratchpad/ppl-decode-kv-17-09.py /tmp/bissect-$c.json 256 1024 2>&1 | grep -iE "ppl|perplex" | tail -1
done
echo "attendu sain ≈ 5,54 ; cassé ≫ 10"
