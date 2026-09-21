#!/usr/bin/env bash
# surveillance-groupe.sh : rapport compact pour la Maîtresse (un passage = une commande).
# carte, processus lourds hors trou, branches récentes non fusionnées, derniers commits par membre, fenêtre de jetons.
cd "$(git rev-parse --show-toplevel)"; date '+== %H:%M'
q=$(cat /tmp/acvram-carte-0.lock.qui 2>/dev/null | head -1); echo "carte: ${q:-libre} ; compute-apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr "\n" " " | cut -c1-120)"
echo "load1: $(cut -d' ' -f1 /proc/loadavg) ; lourds: $(pgrep -af 'pytest|nvcc|ninja|dpkg-deb|dequantiser' | grep -v pgrep | cut -c1-80 | tr '\n' '|' | cut -c1-200)"
git fetch -q origin 2>/dev/null
for b in $(git branch -r --format='%(refname:short)' | grep -E 'origin/(oceane|manon|laurine|laure)'); do
  c=$(git log -1 --format=%ct "$b"); [ "$c" -gt $(( $(date +%s) - 6*3600 )) ] || continue
  git merge-base --is-ancestor "$b" main && f=fusionne || f="NON-fusionne"
  echo "$f ${b#origin/} $(git rev-parse --short "$b") $(git log -1 --format='%H %s' "$b" | cut -c1-8) $(git log -1 --format=%s "$b" | cut -c1-90)"
done
echo "-- verdicts du jour:"; ls -t acvram-memoire/revue/verdict-*$(date +%d-%m)*.md 2>/dev/null | head -6 | xargs -rn1 basename
echo "-- dernier carnet:"; for m in oceane manon laurine laure; do echo "$m: $(grep -n '^## ' acvram-memoire/$m.md 2>/dev/null | tail -1 | cut -c1-70)"; done
echo "-- attentes/accuses (interdits) dans les carnets du jour:"; grep -l -iE "j'attends|en attente|accusé" acvram-memoire/{oceane,manon,laurine,laure}.md 2>/dev/null | tr '\n' ' '; echo
compteur-jetons 2>/dev/null | sed -n '3p'
