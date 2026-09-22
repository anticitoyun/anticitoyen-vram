#!/bin/bash
# Banc horloge SM x energie -- Coder-30B, b=12 (Jerome, suite a Sage §1).
#
# CE SCRIPT NE PREND PAS LE VERROU LUI-MEME : il est concu pour etre
# ENVELOPPE une seule fois, de bout en bout, par `outils/carte.sh` (13/09,
# la campagne tient mieux dans UN verrou continu que dans des verrous par
# etape -- entre un `-lgc` et la mesure qui le vise, un autre appelant
# aurait pu se glisser). L'appeler seul, sans enveloppe, echouera au
# premier acces a outils/gpu/mesure/ (garde du depot).
#
#   ACVRAM_NOM=horloge-laure ACVRAM_TYPE=etat outils/carte.sh \
#       outils/gpu/mesure/banc-horloge-decodage.sh
#
# En service detache (le superviseur de session tue sur MemFree brut, pas
# sur la memoire reellement disponible -- carte.sh:76-84) :
#
#   systemd-run --user --unit=horloge-laure --collect \
#       -p WorkingDirectory="$PWD" \
#       env ACVRAM_NOM=horloge-laure ACVRAM_TYPE=etat \
#       outils/carte.sh outils/gpu/mesure/banc-horloge-decodage.sh
#
# 13/09/2026 : sudoers permet `sudo -n nvidia-smi -lgc/-rgc/-pl` sans mot de
# passe (/etc/sudoers.d/nvidia-smi).
#
# GARDE OBLIGATOIRE : `-rgc` en fin de campagne ET sur toute interruption
# (trap EXIT/INT/TERM) -- une horloge restee verrouillee apres ce script
# affamerait toute session suivante. Le plafond de puissance (400 W) est
# revérifié apres chaque -rgc : ce script ne le TOUCHE jamais (-pl n'est
# appele nulle part ici), seule sa PERSISTANCE apres reset est controlee.
set -u
S="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${ACVRAM_PY:-$S/../../anticitoyen-vram/.venv/bin/python3}
MODEL=$("$(dirname "$0")/../../racine_modeles.py")/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
SCRIPT="$S/outils/gpu/mesure/banc-horloge-decodage.py"
SORTIE_DIR="$S/scratchpad/horloge-decodage-13-09"
mkdir -p "$SORTIE_DIR"

PALIERS_MHZ=(2400 2100 1800 1500)
PLAFOND_ATTENDU_W=400

libre() {
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 800 ] || { echo "REFUS : carte non libre ($u Mio)"; exit 1; }
}

reinit_horloge() {
  # Idempotent : appelable plusieurs fois (trap + fin normale) sans echouer
  # sur une carte deja reinitialisee. PAS de carte.sh ici : le verrou est
  # deja tenu par l'enveloppe exterieure pour toute la duree du script.
  sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
  local pl
  pl=$(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  if [ "$pl" != "$PLAFOND_ATTENDU_W" ]; then
    echo "ALERTE : power.limit=$pl W apres -rgc, attendu $PLAFOND_ATTENDU_W W" >&2
  fi
}
trap reinit_horloge EXIT INT TERM

verrouiller() {
  sudo -n nvidia-smi -i 0 -lgc "${1},${1}"
}

mesurer() {
  local palier="$1" sortie="$2"; shift 2
  "$PY" "$SCRIPT" "$MODEL" "$palier" "$sortie" "$@"
}

echo "=== palier defaut (aucun verrou) ==="
libre
mesurer defaut "$SORTIE_DIR/decodage-defaut.json"
libre
mesurer defaut "$SORTIE_DIR/prefill-defaut.json" --prefill

for f in "${PALIERS_MHZ[@]}"; do
  echo "=== palier ${f} MHz ==="
  libre
  verrouiller "$f" || { echo "REFUS : verrouillage a ${f} MHz echoue"; continue; }
  mesurer "$f" "$SORTIE_DIR/decodage-${f}.json"
done

reinit_horloge
trap - EXIT   # la reinit normale a eu lieu, le trap ne doit pas la repeter

echo
echo "Termine. Resultats dans $SORTIE_DIR/*.json"
nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm,power.limit --format=csv
