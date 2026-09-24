#!/bin/bash
# Pièce 171 (poste6, 25/09) — purge de l'historique du dépôt GitLab, À BLANC par défaut.
#
# Retire de TOUT l'historique, par git-filter-repo, sur un clone miroir de travail :
#   1. les blobs plus gros que SEUIL (défaut 1M : dumps .pt, traces nsys/ncu, logits json) ;
#   2. les deux chemins personnels : « /home/<utilisateur> » → « ~ » et « /tmp/claude-<uid>/-home-… » → « /tmp/claude-session » ;
#   3. les trailers « Co-Authored-By: » et « Claude-Session: » de tous les messages de commit (REGLES § 1).
# Puis écrit un rapport (taille avant/après, retiré, commits réécrits, diff d'arbre de main contre la source).
#
# Idempotent : le miroir est recréé à chaque exécution depuis SOURCE (rien n'est modifié dans le dépôt de travail) ;
# rejoué sur un dépôt déjà purgé, il ne réécrit aucun commit. Ne pousse JAMAIS sans `--pousser` ET PURGE_CONFIRME=oui,
# et seulement en pause complète du circuit, sur ordre du chef : un `push --mirror` réécrit toutes les branches
# (chaque worktree doit ensuite être refait depuis le nouveau main).
#
#   outils/purge-historique.sh                 # à blanc (SOURCE = URL origin du dépôt courant)
#   SOURCE=/chemin/depot outils/purge-historique.sh        # à blanc depuis un clone local (plus rapide)
#   PURGE_CONFIRME=oui outils/purge-historique.sh --pousser  # la vraie purge
set -euo pipefail
DEPOT=${DEPOT:-$(git rev-parse --show-toplevel)}
SOURCE=${SOURCE:-$(git -C "$DEPOT" remote get-url origin)}
TRAVAIL=${TRAVAIL:-${TMPDIR:-/tmp}/purge-historique-acvram}
SEUIL=${SEUIL:-1M}
UTILISATEUR=${UTILISATEUR:-$(basename "$HOME")}
FILTER_REPO=${FILTER_REPO:-$(command -v git-filter-repo || true)}
if [ -z "$FILTER_REPO" ] && [ -x "$TRAVAIL/git-filter-repo" ]; then FILTER_REPO=$TRAVAIL/git-filter-repo; fi
if [ -z "$FILTER_REPO" ]; then
  echo "git-filter-repo absent : FILTER_REPO=/chemin/git-filter-repo (fichier unique : " >&2
  echo "  curl -sfL https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo -o $TRAVAIL/git-filter-repo)" >&2
  exit 64
fi
POUSSER=0; [ "${1:-}" = --pousser ] && POUSSER=1
if [ $POUSSER = 1 ] && [ "${PURGE_CONFIRME:-}" != oui ]; then echo "REFUS : --pousser exige PURGE_CONFIRME=oui (pause complète du circuit, ordre du chef)" >&2; exit 65; fi
mkdir -p "$TRAVAIL"; M=$TRAVAIL/miroir.git; R=$TRAVAIL/rapport.md
SRC_AFFICHEE=$(printf '%s' "$SOURCE" | sed -E 's#//[^@/]*@#//***@#')
taille() { du -sh "$M" | cut -f1; }
compte() { git -C "$M" count-objects -vH | awk '/size-pack/ {print $2, $3}'; }

echo "== 1. miroir de travail depuis $SRC_AFFICHEE (recréé)"
rm -rf "$M"; git clone -q --mirror "$SOURCE" "$M"
MAIN_AVANT=$(git -C "$M" rev-parse refs/heads/main)
N_AVANT=$(git -C "$M" rev-list --all --count); T_AVANT=$(taille); P_AVANT=$(compte)
GROS=$(git -C "$M" rev-list --objects --all | git -C "$M" cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
      | awk -v s="$SEUIL" 'BEGIN{u=substr(s,length(s)); v=substr(s,1,length(s)-1); lim=(u=="M")?v*1048576:(u=="K")?v*1024:s} $1=="blob" && $2>lim {printf "%.1f Mo\t%s\n", $2/1048576, $3}' | sort -rn)
N_GROS=$(printf '%s\n' "$GROS" | grep -c . || true)
N_HOME=$(git -C "$M" log --all -S"/home/$UTILISATEUR" --format=%h | wc -l)
N_TMP=$(git -C "$M" log --all -S'/tmp/claude-' --format=%h | wc -l)
N_TRAILERS=$(git -C "$M" log --all --format=%h --grep='^Co-Authored-By:\|^Claude-Session:' | wc -l)
echo "   commits $N_AVANT, $T_AVANT, pack $P_AVANT ; blobs > $SEUIL : $N_GROS ; commits touchant /home/$UTILISATEUR : $N_HOME, /tmp/claude- : $N_TMP ; trailers : $N_TRAILERS"

echo "== 2. filtre (blobs > $SEUIL, chemins, trailers)"
REMPL=$TRAVAIL/remplacements.txt
printf '%s==>~\nregex:/tmp/claude-[0-9]+/-home-[^/[:space:]]*==>/tmp/claude-session\n' "/home/$UTILISATEUR" > "$REMPL"
CALLBACK='
import re
lignes = message.split(b"\n")
gardees = [l for l in lignes if not re.match(rb"^(Co-Authored-By|Claude-Session):", l, re.I)]
return b"\n".join(gardees).rstrip(b"\n") + b"\n"
'
( cd "$M" && python3 "$FILTER_REPO" --force --strip-blobs-bigger-than "$SEUIL" --replace-text "$REMPL" --message-callback "$CALLBACK" --quiet )
MAIN_APRES=$(git -C "$M" rev-parse refs/heads/main)
N_APRES=$(git -C "$M" rev-list --all --count); T_APRES=$(taille); P_APRES=$(compte)
REECRITS=$(awk 'NR>1 && $1!=$2' "$M/filter-repo/commit-map" | wc -l)
RESTE_GROS=$(git -C "$M" rev-list --objects --all | git -C "$M" cat-file --batch-check='%(objecttype) %(objectsize)' | awk -v s="$SEUIL" 'BEGIN{u=substr(s,length(s)); v=substr(s,1,length(s)-1); lim=(u=="M")?v*1048576:(u=="K")?v*1024:s} $1=="blob" && $2>lim' | wc -l)
RESTE_HOME=$(git -C "$M" log --all -S"/home/$UTILISATEUR" --format=%h | wc -l)
RESTE_TMP=$(git -C "$M" log --all -S'/tmp/claude-' --format=%h | wc -l)
RESTE_TRAILERS=$(git -C "$M" log --all --format=%h --grep='^Co-Authored-By:\|^Claude-Session:' | wc -l)

echo "== 3. preuve d'arbre : main réécrit contre main source"
git -C "$M" fetch -q "$SOURCE" "+refs/heads/main:refs/purge/main-source"
DIFF=$(git -C "$M" diff --stat refs/purge/main-source refs/heads/main | tail -40)
git -C "$M" update-ref -d refs/purge/main-source; git -C "$M" reflog expire --expire=now --all; git -C "$M" gc -q --prune=now

{
  echo "# Purge de l'historique — rapport $( [ $POUSSER = 1 ] && echo 'RÉEL' || echo 'À BLANC' ) ($(date '+%d/%m/%Y %H:%M'))"
  echo "source : $SRC_AFFICHEE · main $MAIN_AVANT → $MAIN_APRES · seuil $SEUIL · filter-repo $(python3 "$FILTER_REPO" --version 2>/dev/null | head -1)"
  echo
  echo "| | avant | après |"; echo "|---|---|---|"
  echo "| commits (toutes refs) | $N_AVANT | $N_APRES (réécrits : $REECRITS) |"
  echo "| taille du miroir | $T_AVANT | $T_APRES |"; echo "| pack | $P_AVANT | $P_APRES |"
  echo "| blobs > $SEUIL | $N_GROS | $RESTE_GROS |"
  echo "| commits touchant /home/$UTILISATEUR | $N_HOME | $RESTE_HOME |"
  echo "| commits touchant /tmp/claude- | $N_TMP | $RESTE_TMP |"
  echo "| commits à trailers Co-Authored-By / Claude-Session | $N_TRAILERS | $RESTE_TRAILERS |"
  echo; echo "## Blobs retirés ($N_GROS)"; printf '%s\n' "$GROS" | sed 's/^/* /'
  echo; echo "## Diff d'arbre main (source → réécrit) — attendu : seulement les blobs retirés et les fichiers dont un chemin a été remplacé"
  echo '```'; printf '%s\n' "$DIFF"; echo '```'
  echo; echo "Remplacements : \`/home/$UTILISATEUR\` → \`~\`, \`/tmp/claude-<uid>/-home-…\` → \`/tmp/claude-session\` (tous les blobs, binaires compris)."
} > "$R"
echo "rapport : $R"; sed -n '1,12p' "$R"

if [ $POUSSER = 1 ]; then
  echo "== 4. POUSSÉE (push --mirror) vers $SRC_AFFICHEE"
  git -C "$M" push --mirror "$SOURCE"
else
  echo "à blanc : rien poussé (ajouter --pousser avec PURGE_CONFIRME=oui, en pause complète, sur ordre du chef)"
fi
