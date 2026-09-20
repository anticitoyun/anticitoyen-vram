#!/bin/bash
# 32 et 16 n'ont jamais tourne. PREDICTION ECRITE AVANT (docs/ATTENTION-
# DECODAGE-ETAT.md) : gain nul ou faible a 32, PERTE a 16 — parce que le
# travail economise (~5 us de 64 a 32) est du meme ordre que la croissance du
# plancher observee en doublant les blocs (~3,8 us de 768 a 1504). Si 32 gagne
# nettement plus, c'est le MODELE ADDITIF qui est faux, pas le reglage.
# poste1 predit l'inverse — encore du gain aux deux — par la capacite lue dans
# le binaire. Aucune de nous deux ne connait les chiffres de l'autre.
#
# ORDRE : 64 encadre le balayage (rejoue au debut ET a la fin). La derive
# thermique deplace l'ordre de passage ; un point de reference aux deux bouts
# la rend visible au lieu de la subir.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$S")"
PY=${ACVRAM_PY:-$R/../../anticitoyen-vram/.venv/bin/python}
M=${ACVRAM_MODELES:-$("$(dirname "$0")/racine_modeles.py")}/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
SORTIE=${SORTIE:-/tmp/poste2-3216}; mkdir -p "$SORTIE"
export PYTHONPATH="$R" CUDA_VISIBLE_DEVICES=0
dire() { echo "[$(date +%H:%M:%S)] $*"; notify-send -a acvram "balayage 32/16" "$*" 2>/dev/null || true; }

for c in 64 32 16 64; do
  for e in 0 3; do
    # ATTENDRE LA TRAINEE, PAS REFUSER DESSUS. Le pilote ne rend pas la VRAM a
    # l'instant ou le processus se termine : le bras suivant demarre sur 985 Mio
    # encore alloues par le precedent et la garde refuse — a juste titre sur
    # l'etat, a tort sur la conclusion. L'attente appartient a l'appelant : un
    # verdict doit rester un verdict. Bornee, sinon un vrai occupant nous
    # ferait tourner sans fin.
    for _ in $(seq 1 30); do
      "$S/carte-libre.sh" 2>/dev/null && break
      sleep 4
    done
    "$S/carte-libre.sh" 2>/dev/null || { dire "REFUS apres 120 s d'attente : $("$S/carte-libre.sh" 2>&1 >/dev/null | head -1)"; exit 1; }
    f="$SORTIE/c$c-e$e-$(date +%s).txt"
    ACVRAM_PA_CHUNK=$c ACVRAM_PA_ETAPE=$e timeout -k 30 1800 \
      $PY -u "$S/attn-isole.py" "$M" 3000 >"$f" 2>&1 \
      || { dire "ECHEC chunk $c etape $e — voir $f"; tail -3 "$f"; exit 1; }
    grep -v '^RESULTAT' "$f" | tail -1
    $PY /tmp/rendre-cache.py "$M" >/dev/null 2>&1   # entre CHAQUE bras
    dire "termine : chunk $c, etape $e"
  done
done
echo; echo "== balayage =="
awk -F'\t' '$1=="RESULTAT"{printf "chunk %5s etape %s  C %3s  %8.2f us [%.2f-%.2f]  part %s\n",$4,$3,$7,$8,$9,$10,$11}' "$SORTIE"/*.txt
dire "BALAYAGE 32/16 TERMINE"
