#!/bin/bash
# LA CARTE SE RESERVE, ELLE NE SE CONSTATE PAS.
#
#     outils/carte.sh <commande...>
#
# `outils/carte-libre.sh` repond « libre a cet instant » — un INSTANTANE, pas
# une reservation. Deux sessions qui l'interrogent a trois secondes d'ecart
# obtiennent toutes deux « oui », et toutes deux ont raison : le 10/09 a 11h05
# deux mesures ont charge en meme temps, l'une est morte en OOM a 3,56 Mio
# libres et l'autre a rendu un chiffre que rien ne signalait comme faux. Aucune
# garde plus fine ne boucherait ce trou : il est de nature, pas de precision.
#
# Ce verrou donne ce que la garde ne peut pas : CELUI QUI ARRIVE SECOND ATTEND.
#
# Trois proprietes voulues :
#   1. l'attente est bornee (defaut 30 min) — remplacer un OOM par un blocage
#      silencieux serait un mauvais echange, on a deja perdu vingt minutes sur
#      un `sync` bloque que rien ne signalait ;
#   2. pendant l'attente on dit QUI tient le verrou et DEPUIS QUAND ;
#   3. le verrou meurt avec le processus, y compris tue par le superviseur —
#      c'est `flock` sur un descripteur qui le garantit, pas un fichier temoin.
set -u
VERROU=${ACVRAM_VERROU:-/tmp/acvram-carte-0.lock}
INFO="$VERROU.qui"
ATTENTE=${ACVRAM_ATTENTE:-1800}
NOM=${ACVRAM_NOM:-$(basename "${1:-mesure}")}
[ $# -ge 1 ] || { echo "usage: carte.sh <commande...>" >&2; exit 64; }

qui_tient() {
  [ -r "$INFO" ] || { echo "detenteur inconnu"; return; }
  read -r p t n < "$INFO" 2>/dev/null || { echo "detenteur inconnu"; return; }
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
    echo "PID $p ($n) depuis $(( $(date +%s) - t )) s"
  else
    # INFO peut survivre a un kill -9 ; le VERROU, lui, est deja libere.
    echo "detenteur disparu (info perimee)"
  fi
}

exec 9>"$VERROU" || { echo "carte.sh : $VERROU inaccessible" >&2; exit 65; }
if ! flock -n 9; then
  echo "carte occupee par $(qui_tient) — attente (max ${ATTENTE} s)" >&2
  debut=$(date +%s)
  while ! flock -w 30 9; do
    ecoule=$(( $(date +%s) - debut ))
    [ "$ecoule" -lt "$ATTENTE" ] || {
      echo "ABANDON apres ${ecoule} s : carte toujours tenue par $(qui_tient)" >&2
      exit 3; }
    echo "  ... $ecoule s, toujours $(qui_tient)" >&2
  done
  echo "carte obtenue apres $(( $(date +%s) - debut )) s" >&2
fi
printf '%s %s %s\n' "$$" "$(date +%s)" "$NOM" > "$INFO"
trap 'rm -f "$INFO"' EXIT
# `9>&-` FERME LE DESCRIPTEUR POUR LA COMMANDE SEULE. Sans lui, l'enfant en
# herite et `flock` ne tombe que quand TOUS les descripteurs sont fermes : tuer
# ce script en -9 laissait alors le verrou tenu par son propre enfant, et le
# suivant attendait jusqu'a l'abandon. Verifie dans les deux sens : sans cette
# fermeture, 25 s d'attente puis echec ; avec, la carte est reprise en 1 s.
# Le shell garde le verrou, la commande ne l'a jamais.
"$@" 9>&-
