#!/bin/bash
# Banc horloge MEMOIRE x SM x energie -- Coder-30B b=12 (chef, duel energie
# du 14/09 : vLLM 0,202 J/jeton contre nos 0,601). Suite au protocole
# revue/protocole-horloge-memoire-decodage-14-09.md.
#
# Meme patron que banc-horloge-decodage.sh (13/09) : ENVELOPPE une seule
# fois par outils/carte.sh, jamais l'inverse. En service detache :
#
#   systemd-run --user --unit=horloge-mem-poste3 --collect \
#       -p WorkingDirectory="$PWD" \
#       env ACVRAM_NOM=horloge-mem-poste3 ACVRAM_TYPE=etat \
#       outils/carte.sh outils/gpu/mesure/banc-horloge-memoire-decodage.sh
#
# 14/09 : `-lmc 15000,15000` a ete ACCEPTE mais ECRETE silencieusement a
# 13801 MHz reel (pas 15000, pas meme 14001 = clocks.max.memory) -- ce
# script relit et publie SYSTEMATIQUEMENT l'horloge REELLE apres chaque
# verrouillage, jamais la valeur demandee (regle 6, CLAUDE.md).
#
# GARDE OBLIGATOIRE : `-rmc` ET `-rgc` en fin de campagne ET sur toute
# interruption (trap EXIT/INT/TERM) -- DEUX horloges a reinitialiser, pas
# une seule comme le banc SM seul.
set -u
S="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${ACVRAM_PY:-$S/../../anticitoyen-vram/.venv/bin/python3}
MODEL=$("$(dirname "$0")/../../racine_modeles.py")/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
SCRIPT="$S/outils/gpu/mesure/banc-horloge-decodage.py"
SORTIE_DIR="$S/scratchpad/horloge-memoire-14-09"
mkdir -p "$SORTIE_DIR"

MEM_MHZ=(14000 12000 10000)
SM_MHZ=(2100)
PLAFOND_ATTENDU_W=400

libre() {
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 800 ] || { echo "REFUS : carte non libre ($u Mio)"; exit 1; }
}

reinit_horloges() {
  # Idempotent (trap + fin normale). Les DEUX resets, dans cet ordre precis
  # n'a pas d'importance (independants), mais toujours les deux.
  sudo -n nvidia-smi -i 0 -rmc >/dev/null 2>&1
  sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
  local pl
  pl=$(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  if [ "$pl" != "$PLAFOND_ATTENDU_W" ]; then
    echo "ALERTE : power.limit=$pl W apres reset, attendu $PLAFOND_ATTENDU_W W" >&2
  fi
}
trap reinit_horloges EXIT INT TERM

verrouiller_mem() { sudo -n nvidia-smi -i 0 -lmc "${1},${1}"; }
verrouiller_sm()  { sudo -n nvidia-smi -i 0 -lgc "${1},${1}"; }

horloge_reelle() {
  # $1 = "mem" ou "sm"
  if [ "$1" = mem ]; then
    nvidia-smi -i 0 --query-gpu=clocks.mem --format=csv,noheader,nounits
  else
    nvidia-smi -i 0 --query-gpu=clocks.sm --format=csv,noheader,nounits
  fi
}

mesurer() {
  local palier="$1" sortie="$2"
  "$PY" "$SCRIPT" "$MODEL" "$palier" "$sortie"
}

echo "=== stock x defaut (temoin, aucun verrou) ==="
libre
mesurer "mem-stock_sm-defaut" "$SORTIE_DIR/mem-stock_sm-defaut.json"

for m in "${MEM_MHZ[@]}"; do
  echo "=== memoire ${m} MHz demande x SM defaut ==="
  libre
  verrouiller_mem "$m" || { echo "REFUS verrou memoire ${m}"; continue; }
  reel=$(horloge_reelle mem)
  echo "  horloge memoire reelle : ${reel} MHz (demande : ${m})"
  mesurer "mem-${m}reel${reel}_sm-defaut" "$SORTIE_DIR/mem-${m}_sm-defaut.json"
  sudo -n nvidia-smi -i 0 -rmc >/dev/null 2>&1

  for f in "${SM_MHZ[@]}"; do
    echo "=== memoire ${m} MHz x SM ${f} MHz ==="
    libre
    verrouiller_mem "$m" || { echo "REFUS verrou memoire ${m}"; continue; }
    verrouiller_sm "$f" || { echo "REFUS verrou SM ${f}"; sudo -n nvidia-smi -i 0 -rmc >/dev/null 2>&1; continue; }
    reel_mem=$(horloge_reelle mem)
    reel_sm=$(horloge_reelle sm)
    echo "  horloges reelles : memoire=${reel_mem} MHz (demande ${m}), SM=${reel_sm} MHz (demande ${f})"
    mesurer "mem-${m}reel${reel_mem}_sm-${f}reel${reel_sm}" \
            "$SORTIE_DIR/mem-${m}_sm-${f}.json"
    sudo -n nvidia-smi -i 0 -rmc >/dev/null 2>&1
    sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
  done
done

echo "=== stock x SM seul (temoin pour comparer a la campagne du 13/09) ==="
for f in "${SM_MHZ[@]}"; do
  libre
  verrouiller_sm "$f" || { echo "REFUS verrou SM ${f}"; continue; }
  reel_sm=$(horloge_reelle sm)
  echo "  SM reel : ${reel_sm} MHz (demande ${f})"
  mesurer "mem-stock_sm-${f}reel${reel_sm}" "$SORTIE_DIR/mem-stock_sm-${f}.json"
  sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
done

reinit_horloges
trap - EXIT

echo
echo "Termine. Resultats dans $SORTIE_DIR/*.json"
nvidia-smi -i 0 --query-gpu=clocks.mem,clocks.max.memory,clocks.sm,clocks.max.sm,power.limit --format=csv
