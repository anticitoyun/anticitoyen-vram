#!/bin/bash
# Quels tests cassent VRAIMENT sous contention, et a quel taux ?
#
# Nous accusons deux tests sur la foi de deux echecs vus au hasard. Un test
# qu'on croit fragile et qui ne l'est pas, on le serialiserait pour rien.
#
# LE CONTROLE SANS LEQUEL LE RESTE NE VAUT RIEN : chaque test est joue N fois
# SANS charge d'abord. S'il echoue deja, ce n'est pas la contention qu'on
# mesure, c'est un test casse — et il est ecarte du tableau, en le disant.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$S")"
PY=~/Bureau/Claude/anticitoyen-vram/.venv/bin/python
export PYTHONPATH="$R" CUDA_VISIBLE_DEVICES=0
N=${N:-5}; GIO=${GIO:-20}; CALCUL=${CALCUL:-0.5}
CIBLES=${CIBLES:-"tests/test_calibration.py tests/test_fusion_nvfp4.py"}

joue() {   # joue <cible> -> "<echecs> <sauts>" sur N manches
  # UN TEST QUI SAUTE N A PAS RESISTE. pytest rend 0 quand tout est ignore :
  # compter le seul code de sortie ferait passer un saut pour un succes — et
  # sous forte charge le garde _carte_disponible fait justement sauter les
  # tests marques. On compte donc les deux, separement.
  local e=0 s=0
  for _ in $(seq 1 $N); do
    local out
    out=$(timeout -k 20 900 $PY -m pytest -q -rs "$1" 2>&1) || e=$((e+1))
    grep -qE '[0-9]+ skipped' <<<"$out" && s=$((s+1))
  done
  echo "$e $s"
}

echo "# N=$N par cible · charge $GIO Gio, calcul $CALCUL"
declare -A sans
for c in $CIBLES; do sans[$c]="$(joue "$c")"; done

# LE VERROU, ET LE NOM QUI DIT QUE LA CHARGE EST VOULUE. Sans lui, une autre
# session voit un PID etranger qui alloue et cherche un intrus : six manches
# perdues le 10/09. `charge-gpu.py` refuse desormais de demarrer sans, donc
# cette ligne n est pas une precaution mais la seule facon de le lancer.
ACVRAM_NOM=CHARGE-DELIBEREE "$S/carte.sh" \
  $PY -u "$S/charge-gpu.py" --gio "$GIO" --calcul "$CALCUL" --duree 3600 &
CH=$!
trap 'kill $CH 2>/dev/null' EXIT
sleep 8

printf '%-42s %8s %8s  %s\n' cible sans avec verdict
for c in $CIBLES; do
  read -r se ss <<<"${sans[$c]}"
  read -r ae as <<<"$(joue "$c")"
  if [ "$se" -gt 0 ]; then
    v="ECARTE : echoue deja $se/$N SANS charge — test casse, pas fragile"
  elif [ "$as" -gt 0 ]; then
    v="ECARTE : a SAUTE $as/$N sous charge — un test qui saute n a pas resiste"
  elif [ "$ae" -eq 0 ]; then
    v="RESISTE $N/$N"
  elif [ "$ae" -eq "$N" ]; then
    v="FRAGILE systematique $N/$N"
  else
    v="FRAGILE intermittent $ae/$N"
  fi
  printf '%-40s %5s/%d %5s/%d  %s\n' "$c" "$se" "$N" "$ae" "$N" "$v"
done
