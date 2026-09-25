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
N_TMP=$(git -C "$M" log --all -G'/tmp/claude-[0-9]+/-home-' --format=%h | wc -l)
REFS_AVANT=$(git -C "$M" for-each-ref --format='%(refname)' | grep -vc '^refs/heads/' || true)
DOLT_AVANT=$(git -C "$M" rev-parse -q --verify refs/dolt/data || echo absent)
N_TRAILERS=$(git -C "$M" log --all --format=%h --grep='^Co-Authored-By:\|^Claude-Session:' | wc -l)
echo "   commits $N_AVANT, $T_AVANT, pack $P_AVANT ; blobs > $SEUIL : $N_GROS ; commits touchant /home/$UTILISATEUR : $N_HOME, /tmp/claude- : $N_TMP ; trailers : $N_TRAILERS"

echo "== 2. filtre (blobs > $SEUIL, chemins, trailers)"
REMPL=$TRAVAIL/remplacements.txt
# `[^/\s]` : classe Python (re), PAS `[:space:]` (POSIX, inopérante ici : la 1re passe à blanc du 25/09 n'a rien remplacé)
printf '%s==>~\nregex:%s==>/tmp/claude-session\n' "/home/$UTILISATEUR" '/tmp/claude-[0-9]+/-home-[^/\s]*' > "$REMPL"
CALLBACK='
import re
lignes = message.split(b"\n")
gardees = [l for l in lignes if not re.match(rb"^(Co-Authored-By|Claude-Session):", l, re.I)]
return b"\n".join(gardees).rstrip(b"\n") + b"\n"
'
# Chemins retirés de TOUT l'historique (chef, 25/09) : les 9 fichiers de scratchpad de poste3 entrés par la fusion c392fe08
# (poussée puis annulée par 0bc30c13), qui portaient des chemins /home/… — retirés plutôt que réécrits.
CHEMINS_RETIRES=(--path scratchpad/poste3-effaceur-23-09 --path scratchpad/poste3-thf-23-09
                 --path-glob 'scratchpad/poste3-piece116-*' --path-glob 'scratchpad/poste3-piece126-*')
# Binaires qui portent le chemin personnel (traces .nsys-rep, .sqlite sous le seuil) : `--replace-text` ne touche pas un blob binaire,
# ils sont retirés de l'historique (chef, 25/09 : absents de main, rien n'est perdu — le contrôle d'arbre le rend faux sinon).
# Lus commit par commit (pickaxe puis grep du blob à ce commit), jamais par extension : un binaire sans le chemin reste.
BINAIRES=$(git -C "$M" log --all -S"/home/$UTILISATEUR" --numstat --format=%H | awk 'NF==1 {c=$1; next} $1=="-" && $2=="-" {print c, $3}' \
           | while read -r c f; do git -C "$M" grep -a -q -e "/home/$UTILISATEUR" "$c" -- "$f" 2>/dev/null && echo "$f"; done | sort -u)
N_BINAIRES=$(printf '%s\n' "$BINAIRES" | grep -c . || true)
for f in $BINAIRES; do CHEMINS_RETIRES+=(--path "$f"); done
# Seules les branches (et tags) sont réécrites : refs/dolt/data (beads) garde ses objets et son hash — `--refs` = mode partiel,
# sans gc ni expiration automatiques (faits plus bas), sans retrait du remote.
REFS=$(git -C "$M" for-each-ref --format='%(refname)' refs/heads refs/tags)
# shellcheck disable=SC2086
( cd "$M" && python3 "$FILTER_REPO" --force --invert-paths "${CHEMINS_RETIRES[@]}" --refs $REFS --quiet )
# shellcheck disable=SC2086
( cd "$M" && python3 "$FILTER_REPO" --force --strip-blobs-bigger-than "$SEUIL" --replace-text "$REMPL" --message-callback "$CALLBACK" --refs $REFS --quiet )
MAIN_APRES=$(git -C "$M" rev-parse refs/heads/main)
N_APRES=$(git -C "$M" rev-list --all --count)
REECRITS=$(awk 'NR>1 && $1!=$2' "$M/filter-repo/commit-map" | wc -l)
RESTE_GROS=$(git -C "$M" rev-list --objects --all | git -C "$M" cat-file --batch-check='%(objecttype) %(objectsize)' | awk -v s="$SEUIL" 'BEGIN{u=substr(s,length(s)); v=substr(s,1,length(s)-1); lim=(u=="M")?v*1048576:(u=="K")?v*1024:s} $1=="blob" && $2>lim' | wc -l)
RESTE_HOME=$(git -C "$M" log --all -S"/home/$UTILISATEUR" --format=%h | wc -l)
RESTE_TMP=$(git -C "$M" log --all -G'/tmp/claude-[0-9]+/-home-' --format=%h | wc -l)
REFS_APRES=$(git -C "$M" for-each-ref --format='%(refname)' | grep -vc '^refs/heads/' || true)
DOLT_APRES=$(git -C "$M" rev-parse -q --verify refs/dolt/data || echo absent)
# commits devenus vides (leur seul contenu était un blob retiré ou un chemin déjà réécrit chez le parent) — lus AVANT le gc
VIDES=$(awk 'NR>1 && $2 ~ /^0+$/ {print $1}' "$M/filter-repo/commit-map" | while read -r h; do git -C "$M" log -1 --format='%h %s' "$h" 2>/dev/null || echo "$h (objet déjà purgé)"; done)
N_VIDES=$(printf '%s\n' "$VIDES" | grep -c . || true)
# binaires sous le seuil qui portent encore le chemin personnel (replace-text ne touche PAS les blobs binaires)
BIN_HOME=$(git -C "$M" grep -a -l "/home/$UTILISATEUR" refs/heads/main 2>/dev/null | sed 's#^refs/heads/main:##' || true)
RESTE_TRAILERS=$(git -C "$M" log --all --format=%h --grep='^Co-Authored-By:\|^Claude-Session:' | wc -l)

echo "== 3. preuve d'arbre : main réécrit contre main source"
git -C "$M" fetch -q "$SOURCE" "+refs/heads/main:refs/purge/main-source"
DIFF=$(git -C "$M" diff --stat refs/purge/main-source refs/heads/main | tail -40)
# Contrôle qui peut rendre faux : les fichiers SUPPRIMÉS de main sont exactement ses blobs > SEUIL (plus les chemins retirés),
# et toute ligne AJOUTÉE dans un fichier modifié porte un remplacement (`~` ou `/tmp/claude-session`).
LIM=$(numfmt --from=iec "$SEUIL")
D_LIST=$(git -C "$M" diff --name-only --diff-filter=D refs/purge/main-source refs/heads/main | sort)
D_ATTENDUS=$(git -C "$M" ls-tree -r -l refs/purge/main-source | awk -v lim="$LIM" '$4!="-" && $4>lim {print $5}' | sort)
D_HORS=$(comm -23 <(printf '%s\n' "$D_LIST") <(printf '%s\n' "$D_ATTENDUS") | grep -vE '^scratchpad/poste3-(effaceur-23-09|thf-23-09|piece116-|piece126-)' || true)
D_MANQUANTS=$(comm -13 <(printf '%s\n' "$D_LIST") <(printf '%s\n' "$D_ATTENDUS") || true)
LIGNES_HORS=$(git -C "$M" diff --diff-filter=M refs/purge/main-source refs/heads/main | grep '^+' | grep -v '^+++' | grep -vc '~\|/tmp/claude-session' || true)
if [ -z "$D_HORS" ] && [ -z "$D_MANQUANTS" ] && [ "$LIGNES_HORS" = 0 ]; then ARBRE="CONFORME"; else ARBRE="NON CONFORME"; fi
git -C "$M" update-ref -d refs/purge/main-source; git -C "$M" reflog expire --expire=now --all; git -C "$M" gc -q --prune=now
T_APRES=$(taille); P_APRES=$(compte)      # après le gc : en mode partiel, filter-repo ne purge pas lui-même les anciens objets

{
  echo "# Purge de l'historique — rapport $( [ $POUSSER = 1 ] && echo 'RÉEL' || echo 'À BLANC' ) ($(date '+%d/%m/%Y %H:%M'))"
  echo "source : $SRC_AFFICHEE · main $MAIN_AVANT → $MAIN_APRES · seuil $SEUIL · filter-repo $(python3 "$FILTER_REPO" --version 2>/dev/null | head -1)"
  echo
  echo "| | avant | après |"; echo "|---|---|---|"
  echo "| commits (toutes refs) | $N_AVANT | $N_APRES (réécrits : $REECRITS) |"
  echo "| taille du miroir | $T_AVANT | $T_APRES |"; echo "| pack | $P_AVANT | $P_APRES |"
  echo "| blobs > $SEUIL | $N_GROS | $RESTE_GROS |"
  echo "| commits touchant /home/$UTILISATEUR | $N_HOME | $RESTE_HOME |"
  echo "| commits touchant /tmp/claude-<uid>/-home-… | $N_TMP | $RESTE_TMP |"
  echo "| refs hors branches (dolt, tags, HEAD) | $REFS_AVANT | $REFS_APRES |"
  echo "| refs/dolt/data | $DOLT_AVANT | $DOLT_APRES |"
  echo "| commits à trailers Co-Authored-By / Claude-Session | $N_TRAILERS | $RESTE_TRAILERS |"
  echo; echo "## Arbre de main : $ARBRE"
  echo "supprimés hors attendus : ${D_HORS:-aucun} · attendus non supprimés : ${D_MANQUANTS:-aucun} · lignes ajoutées sans remplacement : $LIGNES_HORS"
  echo; echo "## Binaires de main qui portent encore /home/$UTILISATEUR (replace-text ne touche pas les binaires) : $(printf '%s\n' "$BIN_HOME" | grep -c . || true)"
  [ -n "$BIN_HOME" ] && printf '%s\n' "$BIN_HOME" | sed 's/^/* /'
  echo; echo "## Binaires retirés de l'historique parce qu'ils portaient /home/$UTILISATEUR ($N_BINAIRES)"; printf '%s\n' "$BINAIRES" | sed 's/^/* /'
  echo; echo "## Commits devenus vides, retirés ($N_VIDES)"; printf '%s\n' "$VIDES" | sed 's/^/* /'
  echo; echo "## Blobs retirés ($N_GROS)"; printf '%s\n' "$GROS" | sed 's/^/* /'
  echo; echo "## Diff d'arbre main (source → réécrit) — attendu : seulement les blobs retirés et les fichiers dont un chemin a été remplacé"
  echo '```'; printf '%s\n' "$DIFF"; echo '```'
  echo; echo "Chemins retirés de l'historique : scratchpad/poste3-effaceur-23-09, poste3-thf-23-09, poste3-piece116-*, poste3-piece126-* (fusion c392fe08, annulée par 0bc30c13)."
  echo "Remplacements : \`/home/$UTILISATEUR\` → \`~\`, \`/tmp/claude-<uid>/-home-…\` → \`/tmp/claude-session\` (blobs texte seulement : filter-repo laisse les binaires)."
  cat <<'PROCEDURE'

## Reprise des postes après la purge RÉELLE (chaque poste, une fois ; le chef d'abord)
La purge réécrit toutes les branches distantes (`push --mirror`) ; `refs/dolt/data` (beads) est poussée telle quelle, son hash ne change pas.
Conditions avant : pause complète du circuit, TOUT poussé (aucun commit local non poussé : ce qui ne l'est pas devra être rebasé à la main).
1. Dépôt principal : `git fetch --prune origin` puis `git reset --hard origin/main` (les anciens commits ne sont plus joignables depuis origin).
2. Chaque worktree (`git worktree list`) : `git -C <worktree> reset --hard origin/<branche>` ; branches locales sans distante (worktree-agent-*,
   branches déjà fusionnées) : `git worktree remove <chemin>` puis `git branch -D <branche>` ; `git worktree prune`.
3. Commit local non poussé sur une branche réécrite : `git rebase --onto origin/<b> <ancien origin/<b>> <b>` (l'ancien sha se lit dans
   `filter-repo/commit-map` du miroir : ancien → nouveau) ; en cas de doute, recloner et réappliquer le diff (`git diff` sauvé avant).
4. Beads : rien à faire (`refs/dolt/data` inchangée, `.beads/` local intact) ; `bd dolt pull` doit être sans effet.
5. Purger les anciens objets locaux : `git reflog expire --expire=now --all && git gc --prune=now` (sinon le 1 G reste sur disque, sans danger).
6. Contrôle : `git rev-list --count --all` = « commits après » du rapport ; `git cat-file -t <ancien sha de main>` échoue après le gc ;
   `git ls-remote origin refs/dolt/data` = le hash du rapport.
7. GitHub public : instantané sans historique, à rejouer par `outils/publier-github.sh --pousser` (rien à rebaser là-bas).
8. Sessions : chaque poste est relancé neuf (`rafraichir-poste`), son worktree refait depuis le nouveau main ; le chef vérifie la reprise.
PROCEDURE
} > "$R"
echo "rapport : $R"; sed -n '1,12p' "$R"

if [ $POUSSER = 1 ]; then
  echo "== 4. POUSSÉE (push --mirror) vers $SRC_AFFICHEE"
  git -C "$M" push --mirror "$SOURCE"
else
  echo "à blanc : rien poussé (ajouter --pousser avec PURGE_CONFIRME=oui, en pause complète, sur ordre du chef)"
fi
