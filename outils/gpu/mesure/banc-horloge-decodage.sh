#!/bin/bash
# Banc horloge SM x energie -- Coder-30B, b=12 (chef, suite a poste7 §1).
#
# CE SCRIPT NE FAIT PAS `sudo` LUI-MEME : verrouiller l'horloge (`nvidia-smi
# -lgc`) exige des privileges que cette session n'a pas et ne doit pas
# obtenir en douce. A CHAQUE PALIER, ce script AFFICHE la commande a taper,
# ATTEND que vous l'ayez fait (Entree pour continuer), puis lance la mesure
# seule (qui, elle, ne touche a rien).
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

libre() {
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 800 ] || { echo "REFUS : carte non libre ($u Mio)"; exit 1; }
}

attendre() {
  echo
  echo ">>> $1"
  read -r -p ">>> Entree une fois fait : "
}

# -- palier "defaut" : pas de verrou, horloge boost normale ------------------
libre
attendre "Aucune commande a taper pour ce palier (horloge par defaut)."
ACVRAM_NOM=horloge-decodage-defaut ACVRAM_TYPE=mesure "$S/outils/carte.sh" \
  "$PY" "$SCRIPT" "$MODEL" defaut "$SORTIE_DIR/decodage-defaut.json"

# -- prefill, horloge haute uniquement (chef : pas de balayage sur ce point)
libre
ACVRAM_NOM=horloge-prefill-defaut ACVRAM_TYPE=mesure "$S/outils/carte.sh" \
  "$PY" "$SCRIPT" "$MODEL" defaut "$SORTIE_DIR/prefill-defaut.json" --prefill

# -- paliers verrouilles -----------------------------------------------------
for f in "${PALIERS_MHZ[@]}"; do
  libre
  attendre "sudo ACVRAM_TYPE=etat $S/outils/carte.sh nvidia-smi -i 0 -lgc ${f},${f}"
  ACVRAM_NOM="horloge-decodage-${f}" ACVRAM_TYPE=mesure "$S/outils/carte.sh" \
    "$PY" "$SCRIPT" "$MODEL" "$f" "$SORTIE_DIR/decodage-${f}.json"
done

# -- reinitialisation de l'horloge, obligatoire --------------------------
attendre "sudo ACVRAM_TYPE=etat $S/outils/carte.sh nvidia-smi -i 0 -rgc"
libre
nvidia-smi -i 0 --query-gpu=clocks.sm,clocks.max.sm --format=csv

echo
echo "Termine. Resultats dans $SORTIE_DIR/*.json"
echo "Comparer jetons_s et j_par_jeton_net entre les paliers (voir la"
echo "prediction scellee dans revue/protocole-horloge-decodage-13-09.md)."
