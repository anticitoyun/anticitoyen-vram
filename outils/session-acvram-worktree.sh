#!/usr/bin/env bash
# Désigne LE worktree « neuf » pour <prénom> : celui de la branche
# origin/<prénom>[-*] la plus récemment COMMITÉE, pas la première ligne de
# `git worktree list | grep -i $p` (pièce 252, chef 26/09 : au redémarrage,
# poste2 et poste4 ont repris sur un vieux worktree — grep renvoie plusieurs
# lignes, dans l'ordre d'AJOUT du worktree, jamais dans l'ordre de fraîcheur
# du travail, et le lanceur prenait la première).
#
# Usage : session-acvram-worktree.sh <prénom minuscule> [racine du dépôt]
# Sortie : le chemin du worktree sur stdout ; code 1 (rien sur stdout) si
# aucune branche origin/<prénom>* n'a de worktree local.
#
# Le prénom doit déjà être en minuscules (le lanceur session-acvram calcule
# `$pl` par iconv+tr avant d'appeler ce script — refaire la translitération
# ici dupliquerait une règle qui doit rester à un seul endroit).
set -euo pipefail
prenom="$1"
racine="${2:-$(pwd)}"
cd "$racine"

# Le chemin du worktree qui a coché LA branche locale "$1" (ex. "poste3-246"),
# ou rien. `--porcelain` : deux lignes consécutives "worktree <chemin>" puis
# "branch refs/heads/<nom>" par worktree — jamais un format qui bouge.
_worktree_de_la_branche() {
  git worktree list --porcelain | awk -v b="refs/heads/$1" '
    $1=="worktree" {p=$2}
    $1=="branch" && $2==b {print p; exit}'
}

# Triées de la plus récemment commitée à la plus ancienne : la PREMIÈRE de
# cette liste qui a un worktree local gagne — jamais besoin de comparer les
# dates nous-mêmes, for-each-ref l'a déjà fait.
while IFS=' ' read -r ref _horodatage; do
  nom="${ref#refs/remotes/origin/}"
  wt="$(_worktree_de_la_branche "$nom")"
  if [ -n "$wt" ]; then
    printf '%s\n' "$wt"
    exit 0
  fi
done < <(git for-each-ref --sort=-committerdate \
            --format='%(refname) %(committerdate:unix)' \
            "refs/remotes/origin/${prenom}" "refs/remotes/origin/${prenom}-*")

exit 1
