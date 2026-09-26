#!/bin/bash
# La carte 0 est-elle prenable ? Contrat, pour tout le circuit :
#     code 0  -> prenable        code 1 -> occupee, stderr dit PAR QUI
#     code 2  -> compilation en cours : LA CARTE EST LIBRE, le processeur est
#                charge. Ni libre ni pris — un troisieme etat, parce que nvcc
#                et ninja ne prennent pas la carte et ne doivent pas bloquer une
#                mesure, mais fausseraient un chiffre sensible au CPU. A
#                l'appelant de decider : `outils/carte-libre.sh; case $? in ...`
#                Une mesure GPU pure peut passer ; un banc de bout en bout non.
# Aucune sortie sur stdout quand c'est libre : appelable en tete de script.
#     outils/carte-libre.sh || exit 1
#
# POURQUOI CE FICHIER EXISTE PLUTOT QUE TROIS COPIES : les deux faux positifs
# ci-dessous ont ete trouves en l'exercant, pas en le subissant. Une copie qui
# ne les porte pas refuse tout ou laisse tout passer, sans le dire.
#
# TROIS CRITERES, DANS CET ORDRE
#   1. VRAM. Necessaire, JAMAIS suffisant : la memoire libre est un ETAT, pas
#      une reservation. Une sonde lancee, precedee d'un `sync`, n'a pas encore
#      alloue — la carte affiche 16 Mio et deux sessions demarrent ensemble.
#   2. Calculs deja alloues sur la carte 0. `-i 0` est OBLIGATOIRE : sans lui,
#      le llama-server permanent du port 8081, qui tourne sur l'AUTRE carte,
#      fait refuser toute mesure pour toujours — et la garde accuse alors un
#      service parfaitement legitime, qu'on cherchera au lieu de la soupconner.
#   3. Intentions annoncees : un python portant un script du depot. Le motif
#      est ancre sur LE DEPOT, pas sur des mots-cles : « profil » attrapait
#      `variety --profile`, le changeur de fond d'ecran du bureau.
#
# ET SES PROPRES ENFANTS ? On remonte les PPID. Sans cela un script s'interdit
# lui-meme des sa deuxieme mesure.
set -u
SEUIL_MIO=${SEUIL_MIO:-800}
# UN INTRUS A DEUX FAÇONS DE NUIRE, et un seuil VRAM n'en voit qu'une. Ce qui
# a invalidé les six manches du 10/09 au soir, ce n'était pas la VRAM de
# `charge-gpu.py`, c'était qu'il CALCULAIT. Et l'inverse : un `acvram serve`
# idle à 1,5 Gio perturbe la mémoire des autres, pas les cycles.
#   non déclaré ET (mem ≥ SEUIL_INTRUS OU sm > 0)   -> INTRUS
#   non déclaré ET mem < seuil ET sm = 0            -> bruit du bureau (Steam)
# Steam : 20 Mio, 0 % sm → bruit. charge-gpu.py : n'importe quelle VRAM, 100 %
# sm → intrus. acvram serve idle : 1,5 Gio, 0 % sm → intrus quand même (mémoire
# contendue). Pas de liste blanche : elle gonfle et ne dit jamais POURQUOI un
# processus est toléré. Le sm% le dit.
SEUIL_INTRUS_MIO=${SEUIL_INTRUS_MIO:-100}
MOTIF=${MOTIF:-'python.*(outils/|acvram)'}
RACINE=${RACINE:-$PPID}          # l'appelant : ses descendants sont « a nous »

# Mes ancetres, sur quelques niveaux seulement. Ils comptent comme « miens » :
# le shell qui a lance la mesure porte la ligne de commande complete dans son
# `bash -c`, donc le motif l'attrape — et sans cette liste la garde se refuse
# elle-meme en accusant son propre appelant. La remontee est BORNEE : plus haut
# on trouverait des ancetres partages avec les autres sessions, et « mien »
# cesserait de vouloir dire quelque chose.
ANCETRES=" "
_p=$$
for _ in 1 2 3 4 5 6; do
  [ "${_p:-0}" -gt 2 ] || break
  ANCETRES="$ANCETRES$_p "
  _p=$(awk '{print $4}' "/proc/$_p/stat" 2>/dev/null) || break
done

mien() {
  local p=$1 n=0
  case "$ANCETRES" in *" $p "*) return 0 ;; esac
  while [ "${p:-0}" -gt 1 ] && [ $n -lt 40 ]; do
    [ "$p" = "$$" ] || [ "$p" = "$RACINE" ] && return 0
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || return 1
    n=$((n + 1))
  done
  return 1
}

u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null) || {
  echo "carte-libre : nvidia-smi muet — on ne conclut pas que c'est libre" >&2; exit 1; }
# Le seuil VRAM total (critère 1) intervient APRÈS le nommage des intrus : sinon
# un intrus non déclaré qui pousse le total au-dessus du seuil fait dire
# « occupée à N Mio » au lieu de « intrus PID X ». Le refus est correct mais le
# message envoie chercher au mauvais endroit — c'est exactement ce que ce
# script existe pour éviter.

# CROISEMENT AVEC TROIS SOURCES. Un verrou seul cache les intrus ; un journal
# seul se laisse abuser par un service tué en -9. La question est « qui répond
# de l'occupation ? » — nvidia-smi dit qui occupe, le verrou dit qui mesure, le
# journal dit qui sert. Un PID GPU présent dans aucun des deux est un INTRUS,
# et c'est le cas qui a fait mal le 10/09 au soir (2373353, 1,5 Gio non
# déclarés). Journal formaté : « PID<TAB>UNITE<TAB>DEPUIS<TAB>PORTS » ; un
# service qui n'y est pas ne compte pas, un PID mort qui y est est purgé
# silencieusement à la lecture (jamais en écriture, un autre process peut
# écrire pendant qu'on lit).
JOURNAL_SERVICES=${JOURNAL_SERVICES:-outils/gpu/journal-services.tsv}
service_declare() {
  local pid=$1
  [ -r "$JOURNAL_SERVICES" ] || return 1
  awk -v p="$pid" -F'\t' '$1 == p { print $2; found=1 } END { exit !found }' \
      "$JOURNAL_SERVICES" 2>/dev/null
}

# DEUX APPELS PARCE QUE `pmon` NE DIT PAS CE QU'ON CROIT. Sa colonne « mem »
# est le POURCENTAGE de bande passante mémoire (trafic), pas l'occupation en
# Mio — vérifié le 11/09 sur un intrus statique de 810 Mio qui rendait sm=0
# mem=0 et passait le critère. Les Mio effectifs viennent de
# --query-compute-apps=used_memory ; pmon garde le sm% qui est ce qu'on
# voulait pour la seconde dimension.
declare -A SM_PCT MEM_MIO
while read -r ligne; do
  case "$ligne" in '#'*|'') continue ;; esac
  set -- $ligne
  [ "${2:-}" = "-" ] && continue
  SM_PCT[$2]=${4:-0}    # $4 = sm% (trafic calcul)
done < <(nvidia-smi pmon -c 1 -i 0 2>/dev/null)
while IFS=', ' read -r pid mem _; do
  [ -n "${pid:-}" ] || continue
  MEM_MIO[$pid]=${mem:-0}
done < <(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)

for pid in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  mien "$pid" && continue
  if service_declare "$pid" >/dev/null; then
    continue          # service déclaré : information, pas refus
  fi
  sm=${SM_PCT[$pid]:-0}; mem=${MEM_MIO[$pid]:-0}
  case "$sm" in ''|*[!0-9]*) sm=0 ;; esac
  case "$mem" in ''|*[!0-9]*) mem=0 ;; esac
  if [ "$sm" = 0 ] && [ "$mem" -lt "$SEUIL_INTRUS_MIO" ]; then
    continue          # bruit du bureau : petit, silencieux, ignoré
  fi
  echo "carte 0 : PID $pid non déclaré, ${mem} Mio, ${sm} % sm — $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-100)" >&2
  echo "  ni verrou (mesure), ni journal (service). Intrus." >&2
  exit 1
done

# Seuil VRAM total en DERNIER recours : après les intrus nommés, il attrape
# les allocations sans processus déclaré (rare mais possible, ex. sonde qui
# alloue avant que nvidia-smi ne la voie).
if [ "$u" -ge "$SEUIL_MIO" ]; then
  echo "carte 0 occupee : $u Mio alloues (seuil $SEUIL_MIO) — aucun intrus nommé" >&2
  exit 1
fi

while read -r pid reste; do
  [ -n "${pid:-}" ] || continue
  mien "$pid" || {
    echo "carte 0 : une mesure demarre sans avoir encore alloue, PID $pid — $(echo "$reste" | cut -c1-120)" >&2
    exit 1; }
done < <(pgrep -af "$MOTIF" 2>/dev/null)

# TROISIEME ETAT. Une compilation est cherchee EN DERNIER, apres que la carte a
# ete declaree libre : c'est une information, pas un refus. La distinction
# compte — classer nvcc avec les mesures aurait fait attendre une compilation
# de 9 minutes qui ne prend pas la carte, et une garde qui fait attendre pour
# rien est une garde qu'on finit par sauter.
# Le motif est ancre sur le NOM DU BINAIRE : un shell dont la ligne de commande
# se contente de MENTIONNER nvcc n'est pas une compilation. Et on ecarte les
# siens, comme partout ailleurs ici.
comp=0
while read -r pid _; do
  [ -n "${pid:-}" ] || continue
  mien "$pid" || comp=$((comp + 1))
done < <(pgrep -af '(^|/)(nvcc|cicc|ptxas|cudafe\+\+|ninja)( |$)' 2>/dev/null)
if [ "$comp" -gt 0 ]; then
  echo "carte 0 libre, mais $comp processus de compilation en cours : le processeur est charge" >&2
  exit 2
fi
exit 0
