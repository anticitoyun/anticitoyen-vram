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
# CE QUE LE VERROU TIENT. Un verrou d'usage exclusif ne voit pas qu'on change
# l'ETAT de la carte : le 10/09, un balayage de frequence a tourne pendant une
# campagne qui respectait pourtant le verrou — 53,67 puis 68,75 pas/s sur le
# meme bras, 28 % d'ecart, parce que l'autre ne PRENAIT pas la carte, il en
# CHANGEAIT l'horloge. Toute modification d'etat (-lgc, -lmc, -pl, mode de
# calcul) prend donc le verrou au meme titre qu'une mesure, et le declare :
# celui qui attend doit savoir s'il attend une manche ou un reglage.
TYPE=${ACVRAM_TYPE:-mesure}
case "$TYPE" in
  mesure|etat) ;;
  *) echo "carte.sh : ACVRAM_TYPE doit valoir 'mesure' ou 'etat'" >&2; exit 64 ;;
esac

# ETAT DE LA CARTE, RELEVE AVANT ET APRES. On ne garantit pas qu'il ne changera
# pas — on CONSTATE apres coup ce qu'on ne pouvait pas garantir avant, comme le
# sha du .so. Un debit se publie avec l'etat de la carte, comme une energie
# avec sa limite de puissance.
# CE QU'ON RELEVE : les PLAFONDS imposes, jamais les frequences instantanees.
# Premier essai fait avec clocks.sm : il s'est declenche a TOUS les coups, la
# carte passant de 225 a 2992 MHz simplement en sortant de veille. Un garde-fou
# qui crie a chaque mesure est un garde-fou qu'on desactive. `-lgc`, `-lmc` et
# `-pl` agissent sur les plafonds — c'est donc eux qui distinguent un reglage
# impose d'une montee en regime normale.
# (clocks.applications.graphics est deprecie sur ce pilote et rend un message
# a la place d'un nombre : il est exclu exprès.)
etat_carte() {
  nvidia-smi -i 0 --query-gpu=clocks.max.sm,clocks.max.mem,power.limit \
             --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
[ $# -ge 1 ] || { echo "usage: carte.sh <commande...>" >&2; exit 64; }

# LE VERROU VIT AUSSI LONGTEMPS QUE LA COMMANDE QU'IL ENVELOPPE — donc cette
# commande doit etre CELLE QUI TRAVAILLE. Un lanceur qui rend la main des que
# le travail est parti (systemd-run, setsid, nohup &, at) fait relacher le
# verrou pendant que la mesure tourne encore : on obtient exactement le cas que
# ce verrou vise, PLUS une fausse assurance. Trouve et verifie par claude-c6
# dans les deux sens : enveloppant systemd-run, le second obtient la carte
# service actif ; carte.sh A L'INTERIEUR du service, le second est refuse.
# Le service detache n'est pas une coquetterie : c'est la voie obligee pour
# echapper au superviseur qui tue sur MemFree. L'ordre est donc
# `systemd-run ... carte.sh <mesure>`, jamais l'inverse.
case "$(basename -- "$1")" in
  systemd-run|setsid|nohup|at|batch|sbatch|screen|tmux|disown)
    cat >&2 <<AVERTISSEMENT
carte.sh : REFUS — « $(basename -- "$1") » rend la main avant la fin du
travail, donc le verrou serait relache pendant que la mesure tourne encore.
Inversez l'ordre : $(basename -- "$1") ... $0 <votre mesure>
Le verrou doit envelopper CE QUI TRAVAILLE, pas ce qui lance.
AVERTISSEMENT
    exit 64 ;;
esac

qui_tient() {
  [ -r "$INFO" ] || { echo "detenteur inconnu"; return; }
  read -r p t n y < "$INFO" 2>/dev/null || { echo "detenteur inconnu"; return; }
  if [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null; then
    echo "PID $p ($n, ${y:-?}) depuis $(( $(date +%s) - t )) s"
  else
    # INFO peut survivre a un kill -9 ; le VERROU, lui, est deja libere.
    echo "detenteur disparu (info perimee)"
  fi
}

# DOUBLE PRISE : le verrou est deja tenu PAR NOUS, plus haut dans la meme
# chaine. Le 10/09/2026, une campagne enveloppee de carte.sh lancait des
# services qui prenaient carte.sh a leur tour : le bras interieur a attendu
# 1800 s le verrou que son propre ancetre tenait, puis a abandonne. Ce n'est
# pas une collision entre sessions, c'est un interblocage avec soi-meme, et
# aucune attente ne le resout. Un descripteur herite ne peut pas etre repris
# par `flock -n`, donc le detecter par marqueur est la seule voie.
if [ -n "${ACVRAM_CARTE_TENUE:-}" ]; then
    echo "carte.sh : REFUS — la carte est DEJA tenue par cette chaine" >&2
    echo "  (PID ${ACVRAM_CARTE_TENUE:-?}, plus haut dans la meme arborescence)." >&2
    echo "  Un seul carte.sh, et c'est le PLUS INTERIEUR qui doit l'avoir :" >&2
    echo "  celui qui execute reellement la mesure. Retirer l'enveloppe" >&2
    echo "  exterieure — attendre ici serait attendre son propre ancetre." >&2
    exit 66
fi
export ACVRAM_CARTE_TENUE=$$
exec 9>"$VERROU" || { echo "carte.sh : $VERROU inaccessible" >&2; exit 65; }
if ! flock -n 9; then
  echo "carte occupee par $(qui_tient) — attente (max ${ATTENTE} s)" >&2
  debut=$(date +%s)
  while :; do
    # LE PAS D'ATTENTE NE DOIT PAS DEPASSER CE QUI RESTE. Avec un `flock -w 30`
    # fixe, une borne plus courte que 30 s n'etait jamais atteinte : le pressé
    # obtenait le verrou au lieu d'abandonner. Trouve en eprouvant l'abandon,
    # qui est justement le cas qu'on ne rencontre jamais par hasard.
    reste=$(( ATTENTE - ($(date +%s) - debut) ))
    [ "$reste" -gt 0 ] || {
      echo "ABANDON apres $(( $(date +%s) - debut )) s : carte toujours tenue par $(qui_tient)" >&2
      exit 3; }
    pas=$(( reste < 30 ? reste : 30 ))
    flock -w "$pas" 9 && break
    echo "  ... $(( $(date +%s) - debut )) s, toujours $(qui_tient)" >&2
  done
  echo "carte obtenue apres $(( $(date +%s) - debut )) s" >&2
fi
printf '%s %s %s %s\n' "$$" "$(date +%s)" "$NOM" "$TYPE" > "$INFO"
trap 'rm -f "$INFO"' EXIT
AVANT=$(etat_carte)
# `9>&-` FERME LE DESCRIPTEUR POUR LA COMMANDE SEULE. Sans lui, l'enfant en
# herite et `flock` ne tombe que quand TOUS les descripteurs sont fermes : tuer
# ce script en -9 laissait alors le verrou tenu par son propre enfant, et le
# suivant attendait jusqu'a l'abandon. Verifie dans les deux sens : sans cette
# fermeture, 25 s d'attente puis echec ; avec, la carte est reprise en 1 s.
# Le shell garde le verrou, la commande ne l'a jamais.
"$@" 9>&-
code=$?
APRES=$(etat_carte)
if [ "$TYPE" = mesure ] && [ -n "$AVANT" ] && [ "$AVANT" != "$APRES" ]; then
  echo "carte.sh : ATTENTION — l'etat de la carte a CHANGE pendant la mesure" >&2
  echo "  avant : $AVANT" >&2
  echo "  apres : $APRES" >&2
  echo "  (clocks.sm, clocks.mem, power.limit) — les valeurs absolues de cette" >&2
  echo "  manche ne sont pas comparables a une autre. Un rapport ABBA mesure" >&2
  echo "  dans une manche unique peut rester valide : les deux bras ont subi" >&2
  echo "  la meme derive. A vous de trancher, en le DISANT." >&2
  [ "$code" -eq 0 ] && code=9
fi
exit $code
