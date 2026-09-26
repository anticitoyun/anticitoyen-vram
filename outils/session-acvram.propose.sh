#!/usr/bin/env bash
# PROPOSITION (pièce 252, poste3, 26/09) — pas appliqué à ~/.local/bin par ce
# fichier, chef le fait lui-même après lecture. Deux écarts corrigés par
# rapport à ~/.local/bin/session-acvram :
#
# (1) « trouve TON worktree » disait `git worktree list | grep -i $p` dans le
# PROMPT envoyé à la session neuve : plusieurs worktrees d'un même prénom
# existent presque toujours (poste3-244, poste3-246, poste3-250...) et grep
# rend la première LIGNE, dans l'ordre d'AJOUT du worktree — jamais dans
# l'ordre de fraîcheur du travail. poste2 et poste4 ont repris sur un vieux
# worktree au redémarrage du 26/09 à cause de ce piège. Remplacé par un
# appel à `outils/session-acvram-worktree.sh` (même pièce), qui classe les
# branches `origin/<prénom>[-*]` par `git for-each-ref --sort=-committerdate`
# et rend le worktree de la plus récente qui en a un.
#
# (2) Le terminal enfant héritait CLAUDE*, ANTHROPIC* et ACVRAM_SESSION de
# la session qui a LANCÉ celui-ci — vu le 26/09 : une relance depuis une
# session désactivait les transcripts (CLAUDE_CODE_CHILD_SESSION faisait
# croire à claude qu'il tournait comme sous-session). Même idiome que
# ~/.config/acvram/chef/relancer-poste.sh (déjà en usage, pas réinventé) :
# purge avant l'export, DANS le shell qui lance claude — gnome-terminal peut
# spawner via son propre serveur D-Bus, l'environnement du script
# session-acvram lui-même n'est pas fiable à purger, seul celui du shell
# final l'est.
p="$1"; u="$2"; m="${3:-sonnet}"; e="${4:-medium}"
if [ "$e" = defaut ] || [ "$e" = default ]; then ef=""; else ef=" --effort \"$e\""; fi
rc=""; [ "${5:-}" = rc ] && rc=" --remote-control '$p'"
pl=$(printf '%s' "$p" | iconv -f utf8 -t ascii//TRANSLIT | tr 'A-Z' 'a-z')
case "$pl" in poste6) herite=" et revue/organisation-22-09.md (ton rôle : porte ÉNERGIE à b=12 — vLLM consomme 7 % de J/jeton en moins à débit égal, pièce 94 ter — et investigations longues quand Opus 5.5 bute ; tes pièces te sont données par chef)";; poste5) herite=" et la fin de poste1.md (ton héritage : moteur, noyaux, algorithmes — tu reprends ses pièces)";; poste1) herite=" et la fin de poste4.md (ton héritage : moteur, noyaux)";; poste2) herite=" et la fin de poste3.md (ton héritage : passes de carte, énergie)";; chef) herite=" et poste8.md (ton héritage : index, menus)";; poste4) herite=" et revue/organisation-22-09.md (ton rôle : questions du groupe posées à duck.ai sur ≥ 3 modèles de raisonnement, file dans ~/Bureau/Vibe/qr.md, synthèses, index)";; poste3) herite=" et la fin de poste9.md et chef.md (ton héritage : conversion, poste, parc, .deb, fusions mécaniques)";; *) herite="";; esac
if [ "$u" = neuf ]; then
  reprise="--name '$p' --model \"$m\"$ef$rc --permission-mode bypassPermissions 'Tu es $p, session acvram (session neuve, l ancienne est irrécupérable). Lis acvram-memoire/revue/ETAT.md (seule entrée), acvram-memoire/$pl.md (fin du carnet)$herite, puis REGLES.md § 1-3 et git log --oneline -n 15. Interdit, REGLES § 6 : aucun cat, grep ni jq .reponse sur corpus/, refus-*.txt ou une sortie de modèle non censuré ; référence par sha256, longueurs par wc -c. Mode caveman obligatoire (~/.claude/rules/caveman.md : réponses et messages les plus courts qui restent compréhensibles). Ensuite, sans attendre : trouve TON worktree (\`outils/session-acvram-worktree.sh $pl\`), fais-y git merge origin/main, lis la fin de TON carnet dans ce worktree (le point de reprise à jour est sur ta branche, pas sur main) et REPRENDS la pièce en cours à l étape notée. Seulement si le carnet n a aucun point de reprise, attends les ordres de chef (session nommée chef) ; tes pointeurs vont à chef.'"
else
  reprise="--resume $u --name '$p' --model \"$m\"$ef$rc --permission-mode bypassPermissions"
fi
if [ -x "$HOME/.local/bin/auto-yes" ]; then WRAP="$HOME/.local/bin/auto-yes"; elif command -v auto-yes >/dev/null 2>&1; then WRAP="auto-yes"; else WRAP=""; fi
WD="$HOME/Bureau/Claude"; [ "$u" = neuf ] && WD="$HOME/Bureau/Claude/anticitoyen-vram"
# PURGE (2) : à insérer en TÊTE de la commande bash -lc existante, avant le
# premier `export` — le reste de cette ligne (export PATH=..., cd, exec
# claude) ne change pas et n'est pas recopié ici (le crochet pre-commit du
# dépôt refuse toute ligne ajoutée qui contient un chemin /home/<mot>/,
# le PATH complet en contient un et n'a aucune raison de changer).
PURGE_ENV='for v in $(env | grep -oE "^(CLAUDE[A-Z_]*|ANTHROPIC[A-Z_]*|ACVRAM_SESSION)=" | tr -d =); do unset "$v"; done; '
exec gnome-terminal --title="$p — acvram ($m/$e)" --working-directory="$WD" -- \
  $WRAP bash -lc "$PURGE_ENV"'export PATH=... CUDA_VISIBLE_DEVICES="" ACVRAM_SESSION='"'$p'"'; cd '"'$WD'"' && "$HOME"/.local/bin/claude '"$reprise"'; exec bash'
# ^ le "..." de PATH= ci-dessus est un raccourci d'affichage : reprendre le
# vrai PATH de la ligne existante (celle du fichier ~/.local/bin/session-acvram,
# pas recopiée dans ce dépôt) tel quel, seul le PURGE_ENV est neuf.
