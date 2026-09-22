#!/bin/bash
# QUEL TEST EN CASSE UN AUTRE ? Bisection mecanique, aucune reflexion.
#
# Si une cible passe seule et tombe en suite, la cause est un ETAT laisse par
# un test precedent : variable d environnement posee et non retiree, memoire
# CUDA non liberee, cache de noyaux reutilise. Aucun de ces coupables ne se
# devine ; tous se lisent une fois le test identifie.
#
#   usage : bisecte-suite.sh <cible>            (ex. tests/test_calibration.py::test_x)
#
# La bisection porte sur les FICHIERS qui precedent la cible dans l ordre de
# collecte de pytest — c est cet ordre-la qui compte, pas l ordre alphabetique.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$S")"
PY=${ACVRAM_PY:-$R/../../anticitoyen-vram/.venv/bin/python}
export PYTHONPATH="$R" CUDA_VISIBLE_DEVICES=0
CIBLE=${1:?usage: bisecte-suite.sh <cible>}
FICHIER=${CIBLE%%::*}

# L ordre de collecte, demande a pytest lui-meme plutot que suppose.
mapfile -t TOUS < <($PY -m pytest --collect-only -q tests/ 2>/dev/null \
  | sed 's/::.*//' | awk '!v[$0]++' | sed '/^$/d')
AVANT=()
for f in "${TOUS[@]}"; do
  [ "$f" = "$FICHIER" ] && break
  AVANT+=("$f")
done
echo "# ${#AVANT[@]} fichiers precedent $FICHIER dans l ordre de collecte"

tombe() {   # tombe <fichiers...> -> 0 si la cible ECHOUE
  timeout -k 30 1800 $PY -m pytest -q -p no:cacheprovider "$@" "$CIBLE" \
    >/dev/null 2>&1 && return 1 || return 0
}

# CONTROLE D ABORD : la cible tombe-t-elle seule ? Si oui, rien a bisecter.
if tombe; then
  echo "ARRET : la cible echoue DEJA SEULE — ce n est pas un effet de suite"
  exit 2
fi
if ! tombe "${AVANT[@]}"; then
  echo "ARRET : la cible PASSE avec tous ses predecesseurs — rien a bisecter,"
  echo "        l effet de suite ne se reproduit pas dans cette manche"
  exit 3
fi

lo=0; hi=${#AVANT[@]}
while [ $((hi - lo)) -gt 1 ]; do
  mid=$(((lo + hi) / 2))
  if tombe "${AVANT[@]:0:$mid}"; then hi=$mid; else lo=$mid; fi
  echo "  ... coupable dans les $hi premiers (pas dans les $lo)" >&2
done
echo "COUPABLE : ${AVANT[$lo]}"
echo "  la cible passe avec les $lo premiers, tombe des qu on ajoute celui-la"
