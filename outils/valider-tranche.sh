#!/bin/bash
# ABBA AU NIVEAU DU PROCESSUS. Le chunk est fixe avant le chargement, sinon le
# graphe CUDA capture l'ancienne grille et le second bras n'existe pas.
#   A = ACVRAM_PA_CHUNK=512 (l'ancien reglage)   B = tranche adaptative
# ABBA et non A puis B : la carte derive en temperature et l'ordre de passage
# fabrique sinon un ecart dans son sens.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$S")"
PY=${ACVRAM_PY:-$R/../../anticitoyen-vram/.venv/bin/python}
M=${M:-${ACVRAM_MODELES:-$("$(dirname "$0")/racine_modeles.py")}/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4}
SORTIE=${SORTIE:-/tmp/poste2-tranche}; mkdir -p "$SORTIE"
export PYTHONPATH="$R" CUDA_VISIBLE_DEVICES=0
LMOTS=${LMOTS:-"350 3000"}; NPASS=${NPASS:-3}
dire() { echo "[$(date +%H:%M:%S)] $*"; notify-send -a acvram "validation tranche" "$*" 2>/dev/null || true; }

for lm in $LMOTS; do
  for m in A B B A; do
    r=512; [ "$m" = "B" ] && r=auto
    n="$SORTIE/l$lm-$m-$(date +%s%N).txt"
    "$S/carte-libre.sh" 2>/dev/null || { dire "REFUS : carte non prenable"; exit 1; }
    timeout -k 30 1800 $PY -u "$S/passage-tranche.py" "$M" "$lm" "$r" "$NPASS" >"$n" 2>&1 \
      || { dire "ECHEC bras $m ($lm mots) — voir $n"; tail -3 "$n"; exit 1; }
    grep '^#' "$n"; dire "termine : $lm mots, bras $m ($r)"
  done
done

echo; echo "== resultat =="
awk -F'\t' '$1=="PASSAGE"{d[$2"\t"$3]=d[$2"\t"$3]" "$4; k[$2"\t"$3]=$7}
END{ for (c in d) printf "%s\tdebits%s\tnoyau %s us\n", c, d[c], k[c] }' \
  "$SORTIE"/*.txt | sort
dire "VALIDATION TERMINEE — $SORTIE"
