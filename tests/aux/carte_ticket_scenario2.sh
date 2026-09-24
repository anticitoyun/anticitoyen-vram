#!/bin/bash
# Pièce 6it, scénario 2 (ordre chef, 24/09) : L + 3 chaîneurs BLOQUÉS
# SIMULTANÉMENT (pas une arrivée fraîche isolée) ; chaque chaîneur se
# remet aussitôt dans la file après sa propre libération, jusqu'à
# N_CYCLES passages au total. Métrique : le RANG de passage de L parmi
# les passages (1 = premier).
set -u
CARTE_ROOT="$1"; VERROU="$2"; DESACTIVE="${3:-0}"; N_CYCLES="${4:-20}"
LOG="$VERROU.ordre"
STOP="$VERROU.stop"
: > "$LOG"
rm -f "$STOP"

# occupant : tient bref, force L et les chaineurs a se bloquer TOUS AVANT sa liberation
( ACVRAM_VERROU="$VERROU" ACVRAM_NOM=occupant ACVRAM_ATTENTE=90 ACVRAM_DUREE_MAX=0 \
  ACVRAM_TYPE=mesure CUDA_VISIBLE_DEVICES="" \
  bash "$CARTE_ROOT/outils/carte.sh" bash -c 'sleep 0.4' ) &
occ_pid=$!
sleep 0.1

lancer() {
  local nom="$1"
  if [ "$DESACTIVE" = 1 ]; then
    ACVRAM_TICKET_DESACTIVE=1 ACVRAM_VERROU="$VERROU" ACVRAM_NOM="$nom" ACVRAM_ATTENTE=90 \
      ACVRAM_DUREE_MAX=0 ACVRAM_TYPE=mesure CUDA_VISIBLE_DEVICES="" \
      bash "$CARTE_ROOT/outils/carte.sh" bash -c "echo $nom >> '$LOG'; sleep 0.03"
  else
    ACVRAM_VERROU="$VERROU" ACVRAM_NOM="$nom" ACVRAM_ATTENTE=90 \
      ACVRAM_DUREE_MAX=0 ACVRAM_TYPE=mesure CUDA_VISIBLE_DEVICES="" \
      bash "$CARTE_ROOT/outils/carte.sh" bash -c "echo $nom >> '$LOG'; sleep 0.03"
  fi
}

# L : une seule tentative, reste bloque tant qu il n a pas gagne
( lancer L ) &
l_pid=$!
sleep 0.15   # L a pris son ticket / entame son attente avant les chaineurs

chaineur() {
  local id="$1" i=0
  while [ ! -f "$STOP" ] && [ "$i" -lt "$N_CYCLES" ]; do
    lancer "C$id"
    i=$((i + 1))
  done
}
chaineur 0 & c0=$!
chaineur 1 & c1=$!
chaineur 2 & c2=$!

t0=$(date +%s)
while :; do
  grep -q "^L$" "$LOG" 2>/dev/null && break
  n=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  [ "$n" -ge $((N_CYCLES + 1)) ] && break
  [ $(( $(date +%s) - t0 )) -ge 15 ] && break
  sleep 0.05
done
touch "$STOP"
wait "$l_pid" "$occ_pid" "$c0" "$c1" "$c2" 2>/dev/null
cat "$LOG"
