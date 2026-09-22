#!/usr/bin/env bash
# publier-github.sh : construit la branche publique « public » (instantané de main, SANS historique,
# sans scratchpad/, acvram-memoire/, .beads/, .claude/) et la pousse sur GitHub seulement avec --pousser.
# Le jeton GitHub vit dans le coffre (~/.config/acvram/jetons.gpg, entrée « github.com ») et n'apparaît jamais ici.
# Refus si un fichier public contient un chemin personnel, un courriel ou un jeton.
set -euo pipefail
DEPOT="$(git rev-parse --show-toplevel)"; cd "$DEPOT"
DIST="https://github.com/anticitoyun/anticitoyen-vram.git"
EXCLUS=(scratchpad acvram-memoire .beads .claude .agents .codex .serena AGENTS.md CLAUDE.md .beads.gate.lock .webui_secret_key acvram_0.6.0_amd64.deb acvram_0.6.32_amd64.deb)
SHA=$(git rev-parse --short main); VERSION=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
git branch -D public >/dev/null 2>&1 || true   # instantané précédent : la branche se reconstruit à chaque passage
W=$(mktemp -d); trap 'git worktree remove --force "$W" 2>/dev/null; git worktree prune' EXIT
git worktree add -q --detach "$W" main
( cd "$W"
  for e in "${EXCLUS[@]}"; do [ -e "$e" ] && git rm -rq --cached "$e" && rm -rf "$e"; done
  echo "== contrôle avant publication"
  # motifs privés ; le détecteur lui-même et les tests qui citent ces motifs comme interdits sont exclus
  MOTIF='/home/anticitoyen|petalmail|glpat-[A-Za-z0-9_-]{15,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
  EXCL=(':!outils/publier-github.sh' ':!tests/test_depot_sans_identite.py' ':!tests/test_paquet_sans_identite.py')
  if git grep -l -E "$MOTIF" -- . "${EXCL[@]}" >/dev/null; then
     echo "REFUS : éléments privés dans l'arbre public :"; git grep -l -E "$MOTIF" -- . "${EXCL[@]}" | head -20; exit 3; fi
  if git log -1 --format=%B | grep -qi 'co-authored-by\|claude-session'; then echo "REFUS : marqueur dans le message"; exit 3; fi
  git checkout -q --orphan public
  git add -A
  git -c user.name=anticitoyen -c user.email=anticitoyen@users.noreply.github.com commit -q -m "acvram ${VERSION} — dépôt public (instantané de main ${SHA}) · soutenir : https://buymeacoffee.com/anticitoyen"
  echo "branche public : $(git rev-parse --short HEAD), $(git ls-files | wc -l) fichiers"
  if [ "${1:-}" = --pousser ]; then
     git -c credential.helper="$HOME/.config/acvram/git-credential-acvram" push -f "$DIST" public:main && echo "poussé sur $DIST (main)"
  else echo "à sec : rien poussé (ajouter --pousser)"; fi
)
