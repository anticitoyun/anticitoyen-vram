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
# LE VERROU NOMME LA CARTE REELLEMENT SERVIE, pas une constante. Le defaut
# etait `carte-0` quelle que soit la carte, et AUCUN script du depot ne le
# surchargeait : une mesure sur la 3080 Ti verrouillait la 5090 qu elle
# n utilisait pas et laissait libre celle qu elle occupait. La garde n etait
# pas fausse — elle protegeait autre chose que ce qu on croyait, et rien ne le
# disait puisqu elle fonctionnait. Elle produisait meme la panne INVERSE de
# celle qu elle previent : deux sessions sur deux cartes se bloquaient, deux
# sessions sur la meme carte passaient.
# ACVRAM_CARTE, PAS CUDA_VISIBLE_DEVICES : les lanceurs de session exportent
# desormais CUDA_VISIBLE_DEVICES="" par defaut (carte invisible tant que rien
# ne la reserve, poste7 14/09) — le lire ici donnerait toujours la carte 0 par
# defaut ("" -> 0 via ${:-0}) sans jamais refleter la carte VOULUE. On prend
# le premier index d'ACVRAM_CARTE, qui est celui que le moteur appelle cuda:0
# une fois exposee plus bas.
_cvd=${ACVRAM_CARTE:-0}; _carte=${_cvd%%,*}
case "${_carte:-0}" in
  ''|*[!0-9]*) _carte=0 ;;          # vide ou non numerique : la carte 0
esac
VERROU=${ACVRAM_VERROU:-/tmp/acvram-carte-$_carte.lock}
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
  mesure|etat|service) ;;
  *) echo "carte.sh : ACVRAM_TYPE doit valoir 'mesure', 'etat' ou 'service'" >&2; exit 64 ;;
esac
# PLAFOND DE PRISE (poste7-tests-30min-20-09 § 3.1, utilisateur 07 h 17 : « aucune prise > 30 min »).
# A l'echeance : SIGTERM a la commande et a ses enfants, 10 s, SIGKILL ; -rgc si un -lgc a ete
# pose par la commande (etat /tmp/acvram-eco-<carte>.json) ; verrou rendu par la sortie ;
# ligne `TIMEOUT` au journal ; code 124. ACVRAM_TYPE=service (les services permanents) exempte.
DUREE_MAX=${ACVRAM_DUREE_MAX:-1800}
[ "$TYPE" = service ] && DUREE_MAX=0
case "$DUREE_MAX" in ''|*[!0-9]*) echo "carte.sh : ACVRAM_DUREE_MAX doit etre un entier de secondes" >&2; exit 64 ;; esac

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
# ce verrou vise, PLUS une fausse assurance. Trouve et verifie par c6
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

# DEUX FICHIERS, DEUX QUESTIONS DIFFERENTES — ET LA SECONDE N'AVAIT PAS DE
# REPONSE JUSQU'AU 10/09.
#
# $INFO repond a « QUI TIENT LA CARTE MAINTENANT » : ecrit par `>`, efface par
# le trap. C'est ce qu'il faut pour qu'un arrivant sache qui attendre.
#
# $JOURNAL repond a « QUI LA TENAIT A 17H53 », et rien n'y repondait. Le
# 10/09, deux mesures jumelles ont montre 10,4 % d'ecart la ou une autre paire
# du meme travail en montrait 0,15 % : une contention averee, une heure
# connue, et AUCUN MOYEN DE SAVOIR QUI. Les manches de cette tranche horaire
# sont restees INDECIDABLES — ni repechables ni condamnables, le pire etat
# pour un resultat.
#
# Ce n'est pas un mecanisme arrete a une dimension : il n'y avait rien a
# etendre, la dimension « historique » n'avait jamais ete posee. `>>`, jamais
# efface, une ligne par prise et une par restitution avec la duree tenue.
JOURNAL="$VERROU.journal"
_pris=$(date +%s)
printf '%s prise   %-8s %-32s %s\n' "$(date +%FT%T)" "$$" "$NOM" "$TYPE" >> "$JOURNAL" 2>/dev/null || true
trap 'rm -f "$INFO"; printf "%s rendue  %-8s %-32s %s tenue=%ss\n" "$(date +%FT%T)" "$$" "$NOM" "$TYPE" "$(( $(date +%s) - _pris ))" >> "$JOURNAL" 2>/dev/null || true' EXIT
AVANT=$(etat_carte)
# `9>&-` FERME LE DESCRIPTEUR POUR LA COMMANDE SEULE. Sans lui, l'enfant en
# herite et `flock` ne tombe que quand TOUS les descripteurs sont fermes : tuer
# ce script en -9 laissait alors le verrou tenu par son propre enfant, et le
# suivant attendait jusqu'a l'abandon. Verifie dans les deux sens : sans cette
# fermeture, 25 s d'attente puis echec ; avec, la carte est reprise en 1 s.
# Le shell garde le verrou, la commande ne l'a jamais.
# LA CARTE SE REND VISIBLE ICI, JAMAIS PLUS TOT. Le verrou tenu, on expose la
# carte reservee (ACVRAM_CARTE, "0" par defaut) a la commande SEULE — pas au
# shell de carte.sh, qui n'en a pas besoin. Une session dont le lanceur exporte
# CUDA_VISIBLE_DEVICES="" ne voit donc la carte qu'a l'INTERIEUR d'un carte.sh,
# jamais avant, jamais par accident dans un sous-processus qui l'aurait heritee.
_TIMEOUT="$VERROU.timeout.$$"
# Lot poste 20/09 : affinite P-cores si ACVRAM_CPUS est pose (ex. 0-15) ; sinon rien ne change.
if [ -n "${ACVRAM_CPUS:-}" ] && command -v taskset >/dev/null; then
  CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" taskset -c "$ACVRAM_CPUS" "$@" 9>&- &
else
  CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" "$@" 9>&- &
fi
_fils=$!
_garde=
if [ "$DUREE_MAX" -gt 0 ]; then
  ( sleep "$DUREE_MAX"; touch "$_TIMEOUT"
    pkill -TERM -P "$_fils" 2>/dev/null; kill -TERM "$_fils" 2>/dev/null; sleep 10
    pkill -KILL -P "$_fils" 2>/dev/null; kill -KILL "$_fils" 2>/dev/null ) 9>&- &
  _garde=$!
fi
wait "$_fils"; code=$?
[ -n "$_garde" ] && { kill "$_garde" 2>/dev/null; wait "$_garde" 2>/dev/null; }
if [ -f "$_TIMEOUT" ]; then
  rm -f "$_TIMEOUT"
  _eco="/tmp/acvram-eco-${ACVRAM_CARTE:-0}.json"
  if [ -f "$_eco" ] && grep -q "\"pid\": *$_fils\b" "$_eco" 2>/dev/null; then
    sudo -n nvidia-smi -i "${ACVRAM_CARTE:-0}" -rgc >/dev/null 2>&1 && rm -f "$_eco" \
      && echo "carte.sh : TIMEOUT — -lgc pose par la commande rendu (-rgc)" >&2
  fi
  printf '%s TIMEOUT %-8s %-32s %s tenue=%ss plafond=%ss\n' "$(date +%FT%T)" "$$" "$NOM" "$TYPE" "$(( $(date +%s) - _pris ))" "$DUREE_MAX" >> "$JOURNAL" 2>/dev/null || true
  echo "carte.sh : TIMEOUT — prise de plus de ${DUREE_MAX}s (ACVRAM_DUREE_MAX), commande tuee, code 124" >&2
  code=124
fi
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
