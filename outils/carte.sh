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
# ne la reserve, Sage 14/09) — le lire ici donnerait toujours la carte 0 par
# defaut ("" -> 0 via ${:-0}) sans jamais refleter la carte VOULUE. On prend
# le premier index d'ACVRAM_CARTE, qui est celui que le moteur appelle cuda:0
# une fois exposee plus bas.
_cvd=${ACVRAM_CARTE:-0}; _carte=${_cvd%%,*}
case "${_carte:-0}" in
  ''|*[!0-9]*) _carte=0 ;;          # vide ou non numerique : la carte 0
esac
VERROU=${ACVRAM_VERROU:-/tmp/acvram-carte-$_carte.lock}
INFO="$VERROU.qui"
# Fichier PARTAGÉ (classe partage) : les prises « carte visible sans calcul »
# (conversions --quant-device cpu…) prennent LOCK_SH ici et coexistent entre
# elles ET avec un service ; une mesure prend LOCK_EX ici EN PLUS de VERROU, ce
# qui l'exclut d'elles. Un service ne touche PAS ce fichier (flux service au bit
# inchangé) : il coexiste naturellement avec les partagés, fichiers disjoints.
SHARE="$VERROU.share"
PROMESSE_MIO=${ACVRAM_PROMESSE_MIO:-512}
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
  mesure|etat|service|partage) ;;
  *) echo "carte.sh : ACVRAM_TYPE doit valoir 'mesure', 'etat', 'service' ou 'partage'" >&2; exit 64 ;;
esac
# PLAFOND DE PRISE (sage-tests-30min-20-09 § 3.1, utilisateur 07 h 17 : « aucune prise > 30 min »).
# A l'echeance : SIGTERM a la commande et a ses enfants, 10 s, SIGKILL ; -rgc si un -lgc a ete
# pose par la commande (etat /tmp/acvram-eco-<carte>.json) ; verrou rendu par la sortie ;
# ligne `TIMEOUT` au journal ; code 124. ACVRAM_TYPE=service (les services permanents) exempte.
DUREE_MAX=${ACVRAM_DUREE_MAX:-1800}
# service ET partage exemptés du plafond : un serveur permanent et une conversion
# longue (parfois > 30 min) sont légitimes.
case "$TYPE" in service|partage) DUREE_MAX=0 ;; esac
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

_lister_partages() {
  # les .qui vivants des prises partagees, pour nommer qui gene une mesure.
  local q pp pn out=""
  for q in "$SHARE".*.qui; do
    [ -e "$q" ] || continue
    read -r pp _ pn _ < "$q" 2>/dev/null
    [ -n "${pp:-}" ] && kill -0 "$pp" 2>/dev/null && out="$out ${pn:-?}(pid $pp)"
  done
  echo "partages tenus :${out:- aucun}"
}

# LA PROMESSE VERIFIEE, PAS SUPPOSEE. Une prise partagee promet : aucun calcul
# GPU lourd (<= ACVRAM_PROMESSE_MIO). A chaque nouvelle prise (n'importe quelle
# classe), on relit compute-apps et on compare au PID de chaque partage declare.
# Violee -> REFUS de LA PRISE EN COURS + ligne PROMESSE-VIOLEE au journal, JAMAIS
# de kill (REGLES : ne jamais tuer un processus GPU inconnu ; l'humain tranche).
_verifier_promesses() {
  local apps q pp pn mem viol=""
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null) || return 0
  for q in "$SHARE".*.qui; do
    [ -e "$q" ] || continue
    read -r pp _ pn _ < "$q" 2>/dev/null
    { [ -n "${pp:-}" ] && kill -0 "$pp" 2>/dev/null; } || continue
    mem=$(printf '%s\n' "$apps" | awk -v pid="$pp" -F', *' '$1==pid{s+=$2} END{print s+0}')
    [ "${mem:-0}" -gt "$PROMESSE_MIO" ] && viol="$viol ${pn:-?}(pid $pp: ${mem} Mio)"
  done
  if [ -n "$viol" ]; then
    printf '%s PROMESSE-VIOLEE %s\n' "$(date +%FT%T)" "$viol" >> "$VERROU.journal" 2>/dev/null || true
    echo "carte.sh : REFUS — promesse partagee violee (> ${PROMESSE_MIO} Mio) :$viol" >&2
    echo "  Aucun kill (REGLES), l'humain tranche ; la prise en cours est refusee." >&2
    exit 5
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

# ── CLASSE PARTAGE : LOCK_SH sur S, coexiste avec partages et service ────────
# La carte est visible mais la commande PROMET de ne pas calculer dessus. Elle
# ne touche PAS VERROU (fd 9) : un service (fd 9) et des partages (fd 8) vivent
# donc cote a cote, fichiers disjoints. Une mesure prend LOCK_EX sur S (plus bas)
# et l'exclut. Pas de plafond (DUREE_MAX=0). .qui par job : SHARE.<pid>.qui.
if [ "$TYPE" = partage ]; then
  exec 8>"$SHARE" || { echo "carte.sh : $SHARE inaccessible" >&2; exit 65; }
  if ! flock -sn 8; then                      # SH echoue seulement si une MESURE tient S en EX
    echo "carte.sh : REFUS — prise 'partage' : une mesure exclusive tient la carte ($(qui_tient))." >&2
    exit 4
  fi
  _verifier_promesses                          # un partage deja la qui trahit sa promesse -> refus + journal
  QP="$SHARE.$$.qui"
  JOURNAL="$VERROU.journal"
  printf '%s %s %s %s\n' "$$" "$(date +%s)" "$NOM" partage > "$QP"
  _pris=$(date +%s)
  printf '%s prise   %-8s %-32s partage\n' "$(date +%FT%T)" "$$" "$NOM" >> "$JOURNAL" 2>/dev/null || true
  trap 'rm -f "$QP"; printf "%s rendue  %-8s %-32s partage tenue=%ss\n" "$(date +%FT%T)" "$$" "$NOM" "$(( $(date +%s) - _pris ))" >> "$JOURNAL" 2>/dev/null || true' EXIT
  if [ -n "${ACVRAM_CPUS:-}" ] && command -v taskset >/dev/null; then
    CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" taskset -c "$ACVRAM_CPUS" "$@" 8>&- &
  else
    CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" "$@" 8>&- &
  fi
  _fils=$!
  wait "$_fils"; exit $?
fi

exec 9>"$VERROU" || { echo "carte.sh : $VERROU inaccessible" >&2; exit 65; }
if ! flock -n 9; then
  # REFUS MUTUEL NOMME service <-> mesure. Un service permanent ne rend jamais
  # le verrou de lui-meme : une mesure qui l'attendrait perdrait ses 30 min pour
  # rien, et un service qui attendrait une mesure bloquerait un serveur. Quand
  # les deux natures s'opposent, on refuse TOUT DE SUITE en nommant le detenteur
  # (code 4), au lieu d'entrer dans l'attente. Deux prises de meme nature, ou un
  # reglage d'etat, attendent normalement : c'est ce que le verrou serialise.
  _dp=""; _dt=""
  [ -r "$INFO" ] && read -r _dp _ _dn _dt < "$INFO" 2>/dev/null
  if [ -n "${_dp:-}" ] && kill -0 "$_dp" 2>/dev/null \
     && { { [ "$TYPE" = service ] && [ "$_dt" = mesure ]; } \
       || { [ "$TYPE" = mesure ] && [ "$_dt" = service ]; }; }; then
    echo "carte.sh : REFUS — prise '$TYPE' demandee, carte tenue par $(qui_tient)." >&2
    echo "  Un service et une mesure ne partagent pas la carte et n'attendent" >&2
    echo "  pas l'un l'autre. Arretez le detenteur (PID $_dp) ou changez de carte." >&2
    exit 4
  fi
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

# ── MESURE / ETAT : une fois P obtenu (donc AUCUNE autre mesure/service ne tient),
# prendre LOCK_EX sur S exclut les PARTAGES. On le fait APRES P : ainsi deux
# mesures se serialisent sur P (attente bornée) au lieu de se refuser sur S, et si
# `flock -n 8` echoue ici c'est forcement qu'un partage tient S (pas une mesure).
# S toujours libre sans partage -> comportement au bit inchange pour l'existant.
if [ "$TYPE" = mesure ] || [ "$TYPE" = etat ]; then
  exec 8>"$SHARE" || { echo "carte.sh : $SHARE inaccessible" >&2; exit 65; }
  if ! flock -n 8; then
    echo "carte.sh : REFUS — prise '$TYPE' : des conversions partagees tiennent la carte." >&2
    echo "  $(_lister_partages). Attendez leur fin ou changez de carte." >&2
    exit 4
  fi
  _verifier_promesses
fi

# ── MODE SERVICE : le serveur DETACHE tient le verrou lui-meme ───────────────
# Un serveur permanent ne peut pas etre la commande synchrone d'un carte.sh : il
# ne rend jamais la main, et l'envelopper bloquerait le lanceur. On le DETACHE
# en lui laissant HERITER le descripteur 9 : tant qu'un processus a ce fd,
# `flock` tient ; a la mort du dernier, il tombe tout seul. C'est l'exact oppose
# du `9>&-` d'une mesure (plus bas). carte.sh sort aussitot ; le lanceur rend la
# main ; le serveur vit seul, verrou compris. Une mesure ne le croira donc plus
# libre (le trou du 21/09 : les lanceurs allouaient en `setsid nohup` sans
# verrou). Un gardien detache — SANS le fd, pour ne pas prolonger le verrou —
# efface le `.qui` a la mort du serveur, pour que `qui_tient()` ne mente pas.
# Le `.qui` garde le format 4 champs `<pid> <epoch> <nom> service` (contrat lu
# par la route /verrou et par qui_tient) : <pid> est celui du SERVEUR.
if [ "$TYPE" = service ]; then
  _log=${ACVRAM_SERVICE_LOG:-/tmp/acvram-service.log}
  _jour="$VERROU.journal"
  if [ -n "${ACVRAM_CPUS:-}" ] && command -v taskset >/dev/null; then
    setsid env CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" taskset -c "$ACVRAM_CPUS" "$@" >> "$_log" 2>&1 < /dev/null &
  else
    setsid env CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" "$@" >> "$_log" 2>&1 < /dev/null &
  fi
  _srv=$!
  printf '%s %s %s %s\n' "$_srv" "$(date +%s)" "$NOM" "$TYPE" > "$INFO"
  printf '%s prise   %-8s %-32s %s (detache, verrou herite)\n' "$(date +%FT%T)" "$_srv" "$NOM" "$TYPE" >> "$_jour" 2>/dev/null || true
  setsid sh -c 'while kill -0 '"$_srv"' 2>/dev/null; do sleep 5; done; rm -f "'"$INFO"'"; printf "%s rendue  %-8s %-32s %s (service mort)\n" "$(date +%FT%T)" "'"$_srv"'" "'"$NOM"'" "'"$TYPE"'" >> "'"$_jour"'" 2>/dev/null' 9>&- >/dev/null 2>&1 < /dev/null &
  echo "$_srv"                                   # le lanceur lit le PID du serveur
  exit 0
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
# `8>&-` en plus de `9>&-` : mesure/etat tiennent aussi S (fd 8) en EX ; la
# commande ne doit hériter ni P ni S, sinon elle prolongerait le verrou après
# notre sortie (meme piege que le fd 9). fd 8 n'existe pas pour un service, mais
# ce chemin n'est atteint que par mesure/etat.
if [ -n "${ACVRAM_CPUS:-}" ] && command -v taskset >/dev/null; then
  CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" taskset -c "$ACVRAM_CPUS" "$@" 8>&- 9>&- &
else
  CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" "$@" 8>&- 9>&- &
fi
_fils=$!
_garde=
if [ "$DUREE_MAX" -gt 0 ]; then
  ( sleep "$DUREE_MAX"; touch "$_TIMEOUT"
    pkill -TERM -P "$_fils" 2>/dev/null; kill -TERM "$_fils" 2>/dev/null; sleep 10
    pkill -KILL -P "$_fils" 2>/dev/null; kill -KILL "$_fils" 2>/dev/null ) 8>&- 9>&- &
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
