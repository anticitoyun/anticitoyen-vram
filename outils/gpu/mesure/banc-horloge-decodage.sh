#!/bin/bash
# Banc horloge SM x energie -- Coder-30B, b=12 (chef, suite a poste7 §1).
#
# 13/09/2026 : sudoers permet desormais `sudo -n nvidia-smi -lgc/-rgc/-pl`
# sans mot de passe (voir /etc/sudoers.d/nvidia-smi) -- ce script verrouille
# et deverrouille lui-meme l'horloge, sous outils/carte.sh (ACVRAM_TYPE=etat).
#
# GARDE OBLIGATOIRE : `-rgc` en fin de campagne ET sur toute interruption
# (trap EXIT/INT/TERM) -- une horloge restee verrouillee apres ce script
# affamerait toute session suivante. Le plafond de puissance (400 W) est
# revérifié apres chaque -rgc : ce script ne le TOUCHE jamais (-pl n'est
# appele nulle part ici), seule sa PERSISTANCE apres reset est controlee.
#
# Usage : outils/gpu/mesure/banc-horloge-decodage.sh
set -u
S="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=~/Bureau/Claude/anticitoyen-vram/.venv/bin/python3
MODEL=/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
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
  # Idempotent : peut etre appele plusieurs fois (trap + fin normale) sans
  # echouer sur une carte deja reinitialisee.
  ACVRAM_NOM=horloge-reset ACVRAM_TYPE=etat "$S/outils/carte.sh" \
    sudo -n nvidia-smi -i 0 -rgc >/dev/null 2>&1
  local pl
  pl=$(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
  if [ "$pl" != "$PLAFOND_ATTENDU_W" ]; then
    echo "ALERTE : power.limit=$pl W apres -rgc, attendu $PLAFOND_ATTENDU_W W" >&2
  fi
}
# CE TRAP EST LA GARDE REELLE : sans lui, un Ctrl-C ou un kill pendant un
# palier laisserait l'horloge verrouillee pour la session suivante — exactement
# le defaut que -rgc en fin de script normal ne couvre pas.
trap reinit_horloge EXIT INT TERM

verrouiller() {
  local f="$1"
  ACVRAM_NOM="horloge-lock-${f}" ACVRAM_TYPE=etat "$S/outils/carte.sh" \
    sudo -n nvidia-smi -i 0 -lgc "${f},${f}"
}

mesurer() {
  local nom="$1" palier="$2" sortie="$3"; shift 3
  ACVRAM_NOM="$nom" ACVRAM_TYPE=mesure "$S/outils/carte.sh" \
    "$PY" "$SCRIPT" "$MODEL" "$palier" "$sortie" "$@"
}

echo "=== palier defaut (aucun verrou) ==="
libre
mesurer horloge-decodage-defaut defaut "$SORTIE_DIR/decodage-defaut.json"
libre
mesurer horloge-prefill-defaut defaut "$SORTIE_DIR/prefill-defaut.json" --prefill

for f in "${PALIERS_MHZ[@]}"; do
  echo "=== palier ${f} MHz ==="
  libre
  verrouiller "$f" || { echo "REFUS : verrouillage a ${f} MHz echoue"; continue; }
  mesurer "horloge-decodage-${f}" "$f" "$SORTIE_DIR/decodage-${f}.json"
done

reinit_horloge
trap - EXIT   # la reinit normale a eu lieu, le trap ne doit pas la repeter

echo
echo "Termine. Resultats dans $SORTIE_DIR/*.json"
nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm,power.limit --format=csv
