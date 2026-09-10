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
TMP=$(mktemp)
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
ACVRAM_NOM=CHARGE-DELIBEREE setsid "$S/carte.sh" \
  $PY -u "$S/charge-gpu.py" --gio "$GIO" --calcul "$CALCUL" --duree 3600 \
  > "$TMP" 2>&1 &
CH=$!
# TUER LE GROUPE, PAS L ENFANT. carte.sh est l enfant, charge-gpu.py le
# PETIT-FILS : un kill sur le premier laissait le second occuper 10 Gio pendant
# la manche suivante — constate, et c est ce qui a fait echouer la charge du
# bras a 27 Gio en OOM.
trap 'kill -- -$CH 2>/dev/null; kill $CH 2>/dev/null' EXIT
sleep 10

# LA CHARGE A-T-ELLE VRAIMENT PRIS ? Sans ce controle, le tableau conclut
# « RESISTE » sur une charge qui a echoue — exactement le defaut que ce
# harnais est cense debusquer chez les autres. Constate le 10/09 : la charge
# de 27 Gio est morte en OOM et le verdict RESISTE 5/5 a ete rendu quand meme.
if ! grep -q '^charge : ' "$TMP"; then
  echo "REFUS : la charge n a pas demarre — rien a conclure sur la contention"
  sed -n '1,6p' "$TMP"
  exit 4
fi
grep '^charge : ' "$TMP"
PRIS=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
MINI=$(python3 -c "print(int($GIO*1024*0.8))")
if [ "$PRIS" -lt "$MINI" ]; then
  echo "REFUS : $PRIS Mio pris pour $GIO Gio demandes — la charge n est pas celle annoncee"
  exit 4
fi

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
