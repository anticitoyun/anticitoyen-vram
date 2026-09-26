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
# anticitoyen-vram-6it (24/09, ordre chef) : file FIFO pour l'attente —
# `flock` seul ne garantit pas l'ordre d'arrivee entre plusieurs attendeurs
# (une attente longue affamee par un flux de prises courtes plus recentes).
# Voir outils/carte-ticket.sh pour le principe (ticket + un seul pretendant
# en vol a la fois vers le verrou reel).
source "$(dirname "${BASH_SOURCE[0]}")/carte-ticket.sh"
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
# PLAFOND DE PRISE (poste7-tests-30min-20-09 § 3.1, utilisateur 07 h 17 : « aucune prise > 30 min »).
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

# anticitoyen-vram-c7w : un `acvram serve` lance par la commande sous
# `setsid` (sa propre session) survit au TERM/KILL de la commande — la 115
# bis de poste2, rendue par TIMEOUT le 23/09 a 23h00, a laisse 28,5 Gio
# occupes et fait tomber la prise suivante de poste5 en OOM. `pkill -P`
# (PID) rate ce cas : un enfant deja reparente (mort intermediaire) ou lance
# depuis un sous-shell n'a plus le PPID attendu au moment du signal. La
# CHAINE, elle, ne bouge pas : `ACVRAM_CARTE_TENUE=$$` (pose plus haut,
# exporte) est herite par TOUT descendant, setsid ou non, jusque dans son
# `/proc/<pid>/environ` — c'est la seule marque qui survit a un
# reparentage. A distinguer de `_verifier_promesses` : la, un processus GPU
# INCONNU n'est jamais tue (l'humain tranche) ; ici, on ne cible QUE nos
# propres descendants prouves par cette chaine, jamais un pid etranger.
_reaper_setsid_orphelins() {
  local moi=$1 journal=$2 nom=$3
  local marque="ACVRAM_CARTE_TENUE=$moi"
  local pids p vu
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || return 0
  for p in $pids; do
    p=${p//[[:space:]]/}
    [ -n "$p" ] && [ -r "/proc/$p/environ" ] || continue
    vu=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -Fx "$marque") || continue
    [ -n "$vu" ] || continue
    printf '%s ORPHELIN %-8s %-32s mesure setsid pid %s TERM (c7w)\n' "$(date +%FT%T)" "$moi" "$nom" "$p" >> "$journal" 2>/dev/null || true
    kill -TERM "$p" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    if kill -0 "$p" 2>/dev/null; then
      kill -KILL "$p" 2>/dev/null
      printf '%s ORPHELIN %-8s %-32s mesure setsid pid %s KILL (c7w)\n' "$(date +%FT%T)" "$moi" "$nom" "$p" >> "$journal" 2>/dev/null || true
    fi
  done
}

# Piece 244 : un TERM/INT/HUP recu par carte.sh LUI-MEME (un `kill` externe,
# pas notre propre garde DUREE_MAX) tuait ce script sans jamais toucher a la
# commande, encore vivante dans son groupe (`setsid`, ci-dessous) — le verrou
# tombe (flock lie au descripteur de CE process), la commande continue hors
# verrou. Signal transmis au GROUPE de la commande (pgid negatif : elle-meme
# et tout ce qu'elle a lance), TERM puis KILL apres un repit, avant de sortir
# — `exit` depuis un trap de signal declenche quand meme le trap EXIT plus
# bas (verrou rendu, journal, reaper c7w).
_244_signal() {
  local sig=$1
  printf '%s SIGNAL  %-8s %-32s %s recu %s, transmis au groupe\n' "$(date +%FT%T)" "$$" "${NOM:-?}" "${TYPE:-?}" "$sig" >> "${JOURNAL:-/dev/null}" 2>/dev/null || true
  if [ -n "${_fils:-}" ]; then
    kill -TERM -- "-$_fils" 2>/dev/null
    for _ in 1 2 3 4 5; do kill -0 -- "-$_fils" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$_fils" 2>/dev/null
  fi
  exit $(( 128 + sig ))
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
  # anticitoyen-vram-6it : ticket d'abord (ordre d'arrivee), puis seulement
  # le tour venu, tentative du verrou reel — voir carte-ticket.sh.
  # ACVRAM_TICKET_DESACTIVE : bras cassant / secours, saute le ticket et
  # retombe sur le flock nu d'avant (l'ordre d'arrivee n'est alors plus
  # garanti — sert au test qui PROUVE le defaut sur ce chemin).
  if [ -z "${ACVRAM_TICKET_DESACTIVE:-}" ]; then
    _mon_ticket=$(_ticket_prendre "$VERROU" 12)
    while :; do
      reste=$(( ATTENTE - ($(date +%s) - debut) ))
      [ "$reste" -gt 0 ] || {
        echo "ABANDON apres $(( $(date +%s) - debut )) s : carte toujours tenue par $(qui_tient)" >&2
        exit 3; }
      _ticket_attendre_son_tour "$VERROU" "$_mon_ticket" "$reste" 13 && break
      echo "  ... $(( $(date +%s) - debut )) s (ticket $_mon_ticket), toujours $(qui_tient)" >&2
    done
    echo "$$" > "$VERROU.servi_pid" 2>/dev/null || true
  fi
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
  [ -z "${ACVRAM_TICKET_DESACTIVE:-}" ] && _ticket_avancer "$VERROU" "$_mon_ticket" 12
  echo "carte obtenue apres $(( $(date +%s) - debut )) s" >&2
fi

# Piece 244 : le verrou P vient d'etre obtenu, mais un `.qui` d'une prise
# precedente peut rester si son proprietaire a ete tue en KILL -9 (aucun trap
# ne rattrape -9 : ni l'ancien `_reaper_setsid_orphelins`, ni le _244_signal
# ci-dessus n'ont pu tourner). 5e champ = pgid de la commande (voir plus bas) ;
# un pgid encore vivant, verrou libre, est le signe exact du defaut du 26/09 —
# la carte est REFUSEE plutot que rendue a l'aveugle par-dessus une commande
# qui tourne encore.
_ancien_p=; _ancien_pgid=
read -r _ancien_p _ _ _ _ancien_pgid 2>/dev/null < "$INFO" || true
if [ -n "${_ancien_pgid:-}" ] && kill -0 -- "-$_ancien_pgid" 2>/dev/null; then
  echo "carte.sh : REFUS — le groupe pgid $_ancien_pgid d'une prise precedente ($INFO) est" >&2
  echo "  encore vivant alors que le verrou est libre (KILL -9 du carte.sh parent, sans doute) :" >&2
  echo "  achevez-le (kill -- -$_ancien_pgid) avant de reprendre la carte." >&2
  exit 4
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
  # anticitoyen-vram-jxm : entre la mort du service et ce reveil (jusqu'a 5 s
  # de sommeil), un NOUVEAU detenteur (mesure ou autre service) peut deja avoir
  # pris le verrou libere et ecrit SON .qui a la meme adresse ($INFO, partagee
  # par carte) — un `rm -f` inconditionnel effacait alors l'INFO du detenteur
  # ACTUEL, encore vivant, flock compris (qui_tient() mentait « personne »).
  # Ne retirer .qui que s'il porte encore CE pid de service.
  setsid sh -c 'while kill -0 '"$_srv"' 2>/dev/null; do sleep 5; done; p=; read -r p _ < "'"$INFO"'" 2>/dev/null; [ "$p" = "'"$_srv"'" ] && rm -f "'"$INFO"'"; printf "%s rendue  %-8s %-32s %s (service mort)\n" "$(date +%FT%T)" "'"$_srv"'" "'"$NOM"'" "'"$TYPE"'" >> "'"$_jour"'" 2>/dev/null' 9>&- >/dev/null 2>&1 < /dev/null &
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
# anticitoyen-vram-jxm (ticket resté ouvert : deux .qui de service ont disparu
# sans aucune prise/rendue au journal entre-temps — ce trap n'explique PAS ces
# deux cas, mais porte la MEME classe de defaut que le gardien de service
# (rm -f "$INFO" inconditionnel) : meme garde par symetrie (defense en
# profondeur, flock devrait deja l'empecher), et une trace ANOMALIE si jamais
# ce trap trouvait un .qui qui n'est plus le sien — pour laisser une preuve la
# prochaine fois, au lieu d'un silence.
trap '[ -n "${_charge_pid:-}" ] && kill "$_charge_pid" 2>/dev/null; _reaper_setsid_orphelins "$$" "$JOURNAL" "$NOM"; p=; read -r p _ < "$INFO" 2>/dev/null; if [ "$p" = "$$" ]; then rm -f "$INFO"; printf "%s rendue  %-8s %-32s %s tenue=%ss\n" "$(date +%FT%T)" "$$" "$NOM" "$TYPE" "$(( $(date +%s) - _pris ))" >> "$JOURNAL" 2>/dev/null || true; else printf "%s ANOMALIE %-8s %-32s %s .qui deja repris par pid %s (jxm), non efface\n" "$(date +%FT%T)" "$$" "$NOM" "$TYPE" "${p:-?}" >> "$JOURNAL" 2>/dev/null || true; fi' EXIT
# Piece 244 : voir _244_signal ci-dessus — un TERM/INT/HUP externe sur carte.sh
# ne doit jamais laisser la commande vivre hors verrou.
trap '_244_signal 15' TERM
trap '_244_signal 2' INT
trap '_244_signal 1' HUP
AVANT=$(etat_carte)

# CHARGE HOTE, PENDANT LA PRISE (189, ordre chef, apres la 186 : « une mesure
# doit porter sa propre preuve de charge hote ») — sans elle, un A/B fausse par
# une charge externe reste indecidable, comme les 341 ms de la 186 avant l'ABBA.
# 1 Hz, /proc/loadavg seul (charge globale du systeme, y compris les processus
# hors carte.sh) : suffisant pour dater une contention a la minute pres au
# croisement avec le journal ; vmstat aurait ajoute une dependance externe pour
# le meme besoin. `mesure` SEULEMENT (etat/partage/service ne calculent pas ou
# echappent au plafond) ; arret garanti par le trap EXIT ci-dessus, meme sur
# TIMEOUT ou signal — jamais laisse tourner apres la restitution du verrou.
if [ "$TYPE" = mesure ]; then
  CHARGE_DIR="${ACVRAM_CHARGE_DIR:-$HOME/.cache/acvram/charge}"
  mkdir -p "$CHARGE_DIR" 2>/dev/null
  _nom_fs=$(printf '%s' "$NOM" | tr -c 'A-Za-z0-9_.-' '_')
  CHARGE_FICHIER="$CHARGE_DIR/$_nom_fs.$$.tsv"
  ( while :; do
      printf '%s\t%s\n' "$(date +%FT%T)" "$(cut -d' ' -f1-3 /proc/loadavg)" >> "$CHARGE_FICHIER" 2>/dev/null
      sleep 1
    done ) 8>&- 9>&- </dev/null >/dev/null 2>&1 &
  _charge_pid=$!
fi
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
# Piece 244 (chef, contamination du 26/09 : un pytest orphelin de la commande
# a tourne 10 min hors verrou apres qu'un `kill` externe a tue carte.sh lui-meme
# — 3e cas en trois jours). `setsid` fait de la commande le chef de son PROPRE
# groupe de processus (pgid = son pid) : un signal envoye a `-$_fils` (pgid
# negatif) atteint la commande ET tout ce qu'elle a lance, meme deja reparente,
# sans dependre de `nvidia-smi` (le reaper c7w ci-dessus ne voit que ce qui
# touche deja le GPU — un pytest encore sur CPU lui echappe entierement).
if [ -n "${ACVRAM_CPUS:-}" ] && command -v taskset >/dev/null; then
  setsid env CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" taskset -c "$ACVRAM_CPUS" "$@" 8>&- 9>&- &
else
  setsid env CUDA_VISIBLE_DEVICES="${ACVRAM_CARTE:-0}" "$@" 8>&- 9>&- &
fi
_fils=$!
# 5e champ, en plus du format 4 champs historique de $INFO (lu par qui_tient()
# et /verrou qui n'en lisent que 4) : le pgid de la commande, pour que la
# PROCHAINE prise puisse verifier qu'il n'en reste rien avant de continuer.
# N'ECRASE PAS un $INFO deja repris par quelqu'un d'autre entre-temps (meme
# garde que le trap EXIT generique, jxm) : si le premier champ n'est plus $$,
# ce n'est plus notre fichier a completer.
_p5=; read -r _p5 _ 2>/dev/null < "$INFO"
[ "$_p5" = "$$" ] && printf '%s %s %s %s %s\n' "$$" "$_pris" "$NOM" "$TYPE" "$_fils" > "$INFO"
_garde=
if [ "$DUREE_MAX" -gt 0 ]; then
  # anticitoyen-vram-575 : ce garde-fou heritait stdout/stderr de carte.sh (donc
  # de l'appelant) sans jamais les fermer — seuls 8/9 l'etaient. Un appelant qui
  # CAPTURE la sortie (`subprocess.run(capture_output=True)`, un pipe) attend
  # l'EOF des DEUX tuyaux ; celui du garde ne venait qu'a l'expiration de
  # DUREE_MAX (1800 s par defaut), meme quand la commande finissait en 1 s. Pire :
  # `kill "$_garde"` plus bas ne tue que la sous-shell, pas le `sleep` qu'elle a
  # lance en premier plan — sans `pkill -P`, ce sleep devient orphelin et garde
  # les descripteurs heures apres. Deux correctifs : fermer 1/2 ici (l'appelant
  # ne bloque plus, meme si le nettoyage rate), et tuer explicitement les
  # enfants du garde a la sortie (plus d'orphelin).
  ( sleep "$DUREE_MAX"; touch "$_TIMEOUT"
    pkill -TERM -P "$_fils" 2>/dev/null; kill -TERM "$_fils" 2>/dev/null; sleep 10
    pkill -KILL -P "$_fils" 2>/dev/null; kill -KILL "$_fils" 2>/dev/null
  ) 8>&- 9>&- </dev/null >/dev/null 2>&1 &
  _garde=$!
fi
wait "$_fils"; code=$?
[ -n "$_garde" ] && { pkill -TERM -P "$_garde" 2>/dev/null; kill "$_garde" 2>/dev/null; wait "$_garde" 2>/dev/null; }
if [ -f "$_TIMEOUT" ]; then
  rm -f "$_TIMEOUT"
  _eco="/tmp/acvram-eco-${ACVRAM_CARTE:-0}.json"
  if [ -f "$_eco" ] && grep -q "\"pid\": *$_fils\b" "$_eco" 2>/dev/null; then
    sudo -n nvidia-smi -i "${ACVRAM_CARTE:-0}" -rgc >/dev/null 2>&1 && rm -f "$_eco" \
      && echo "carte.sh : TIMEOUT — -lgc pose par la commande rendu (-rgc)" >&2
  fi
  # c7w : reperer et achever les setsid orphelins de LA COMMANDE tuee, AVANT
  # de journaliser TIMEOUT — "et seulement ensuite rendre" (le rendu reel se
  # fait dans le trap EXIT plus bas, mais la memoire doit etre libre ici deja,
  # la prise suivante peut demarrer des la sortie de ce script).
  _reaper_setsid_orphelins "$$" "$JOURNAL" "$NOM"
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
