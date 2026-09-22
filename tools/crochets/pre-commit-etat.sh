#!/usr/bin/env bash
# Crochet pre-commit (Sage, sage-reorganisation-jetons-pause-20-09) : ETAT.md ≤ 12 000 octets,
# sinon le commit est refusé — ETAT est relu par cinq sessions à chaque relance.
# Installation : cp tools/crochets/pre-commit-etat.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
f=acvram-memoire/revue/ETAT.md; max=12000
if git diff --cached --name-only | grep -qx "$f"; then
  n=$(git show ":$f" | wc -c)
  if [ "$n" -gt "$max" ]; then
    echo "REFUS : $f pèse $n o > $max o — déplacer les têtes datées dans revue/etat-archive-<date>.md" >&2
    exit 1
  fi
fi
exit 0
