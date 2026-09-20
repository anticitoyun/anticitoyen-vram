#!/bin/bash
# Nettoyage des modeles en trois phases REVERSIBLES (poste7-nettoyage-modeles-20-09 § 0) :
#   nettoyage-modeles.sh LISTE              deplace chaque chemin de LISTE vers <disque>/corbeille-<date>/
#                                           (meme systeme de fichiers : instantane), journal.tsv dans la corbeille
#   nettoyage-modeles.sh --restaurer CORB   remet chaque entree du journal a sa place
#   nettoyage-modeles.sh --purger CORB --oui efface la corbeille (irreversible, apres le mot de l'utilisateur)
# Refus : chemin absent, chemin hors des disques autorises (ACVRAM_NETTOYAGE_RACINES, defaut les deux disques
# de modeles), fichier ouvert (fuser), prise de carte en cours (IO pendant une mesure) sauf
# ACVRAM_NETTOYAGE_PENDANT_MESURE=1. Un chemin refuse n'arrete pas la liste : il est journalise « REFUS ».
set -u
RACINES=${ACVRAM_NETTOYAGE_RACINES:-/mnt/2TO_2023_980PRO:/mnt/4TO_SATACMR_2022}
VERROU_QUI=${ACVRAM_VERROU:-/tmp/acvram-carte-0.lock}.qui
DATE=$(date +%Y-%m-%d)

racine_de() {  # la racine autorisee qui contient $1, ou rien
  local p=$1 r; IFS=: read -ra rs <<< "$RACINES"
  for r in "${rs[@]}"; do case "$p" in "$r"/*) echo "$r"; return 0 ;; esac; done
  return 1
}

phase_deplacer() {
  local liste=$1 n=0 refus=0
  [ -r "$liste" ] || { echo "liste illisible : $liste" >&2; exit 64; }
  if [ -e "$VERROU_QUI" ] && [ "${ACVRAM_NETTOYAGE_PENDANT_MESURE:-0}" != 1 ]; then
    echo "prise de carte en cours ($VERROU_QUI) : pas d'IO disque pendant une mesure" >&2; exit 75
  fi
  while IFS= read -r ligne; do
    ligne=${ligne%%#*}; ligne=${ligne%"${ligne##*[! ]}"}; [ -z "$ligne" ] && continue
    local p=$ligne r corb dest taille
    if [ ! -e "$p" ] && [ ! -L "$p" ]; then echo "REFUS absent : $p"; refus=$((refus+1)); continue; fi
    r=$(racine_de "$p") || { echo "REFUS hors des racines autorisees : $p"; refus=$((refus+1)); continue; }
    if [ ! -L "$p" ] && fuser -s "$p" 2>/dev/null; then echo "REFUS fichier ouvert : $p"; refus=$((refus+1)); continue; fi
    corb="$r/corbeille-$DATE"; dest="$corb/${p#"$r"/}"
    mkdir -p "$(dirname "$dest")" || { echo "REFUS mkdir : $dest"; refus=$((refus+1)); continue; }
    taille=$(du -sk "$p" 2>/dev/null | cut -f1)
    if mv -n "$p" "$dest"; then
      printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$p" "$dest" "${taille:-0}" >> "$corb/journal.tsv"
      echo "DEPLACE ${taille:-0}K : $p -> $dest"; n=$((n+1))
    else echo "REFUS mv : $p"; refus=$((refus+1)); fi
  done < "$liste"
  echo "deplaces=$n refus=$refus"
  [ "$refus" -eq 0 ]
}

phase_restaurer() {
  local corb=$1 n=0
  [ -r "$corb/journal.tsv" ] || { echo "pas de journal dans $corb" >&2; exit 64; }
  while IFS=$'\t' read -r quand src dest taille; do
    [ -e "$dest" ] || [ -L "$dest" ] || { echo "REFUS absent de la corbeille : $dest"; continue; }
    [ -e "$src" ] && { echo "REFUS destination occupee : $src"; continue; }
    mkdir -p "$(dirname "$src")" && mv -n "$dest" "$src" && { echo "RESTAURE : $src"; n=$((n+1)); }
  done < "$corb/journal.tsv"
  echo "restaures=$n"
}

phase_purger() {
  local corb=$1 oui=${2:-}
  [ -d "$corb" ] || { echo "corbeille absente : $corb" >&2; exit 64; }
  case "$corb" in */corbeille-????-??-??) ;; *) echo "purge refusee : $corb n'est pas une corbeille datee" >&2; exit 64 ;; esac
  du -sh "$corb"; [ -r "$corb/journal.tsv" ] && wc -l < "$corb/journal.tsv" | sed 's/^/entrees : /'
  [ "$oui" = --oui ] || { echo "purge non faite : ajouter --oui apres le mot de l'utilisateur"; exit 1; }
  rm -rf -- "$corb" && echo "PURGE : $corb"
}

case "${1:-}" in
  --restaurer) phase_restaurer "${2:?corbeille}" ;;
  --purger)    phase_purger "${2:?corbeille}" "${3:-}" ;;
  ""|-h|--help) sed -n '2,10p' "$0" ;;
  *)           phase_deplacer "$1" ;;
esac
