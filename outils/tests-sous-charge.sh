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

joue() {   # joue <cible> -> nombre d echecs sur N
  local e=0
  for _ in $(seq 1 $N); do
    timeout -k 20 900 $PY -m pytest -q "$1" >/dev/null 2>&1 || e=$((e+1))
  done
  echo $e
}

echo "# N=$N par cible · charge $GIO Gio, calcul $CALCUL"
declare -A sans
for c in $CIBLES; do sans[$c]=$(joue "$c"); done

$PY -u "$S/charge-gpu.py" --gio "$GIO" --calcul "$CALCUL" --duree 3600 &
CH=$!
trap 'kill $CH 2>/dev/null' EXIT
sleep 8

printf '%-42s %8s %8s  %s\n' cible sans avec verdict
for c in $CIBLES; do
  a=$(joue "$c")
  if [ "${sans[$c]}" -gt 0 ]; then
    v="ECARTE : echoue deja $((${sans[$c]}))/$N SANS charge"
  elif [ "$a" -eq 0 ]; then
    v="resiste"
  elif [ "$a" -eq "$N" ]; then
    v="FRAGILE systematique"
  else
    v="FRAGILE intermittent $a/$N"
  fi
  printf '%-42s %6s/%d %6s/%d  %s\n' "$c" "${sans[$c]}" "$N" "$a" "$N" "$v"
done
