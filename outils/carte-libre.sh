#!/bin/bash
# La carte 0 est-elle prenable ? Contrat, pour tout le circuit :
#     code 0  -> prenable        code 1 -> occupee, stderr dit PAR QUI
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
if [ "$u" -ge "$SEUIL_MIO" ]; then
  echo "carte 0 occupee : $u Mio alloues (seuil $SEUIL_MIO)" >&2; exit 1
fi

for pid in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  mien "$pid" || {
    echo "carte 0 : calcul en cours, PID $pid — $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-120)" >&2
    exit 1; }
done

while read -r pid reste; do
  [ -n "${pid:-}" ] || continue
  mien "$pid" || {
    echo "carte 0 : une mesure demarre sans avoir encore alloue, PID $pid — $(echo "$reste" | cut -c1-120)" >&2
    exit 1; }
done < <(pgrep -af "$MOTIF" 2>/dev/null)
exit 0
