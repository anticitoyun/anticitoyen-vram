#!/usr/bin/env bash
# sortir-version.sh vX.Y.Z [--simule]
#
# Pièce 281 (chef, 26/09) : codifie la sortie de version faite à la main jusqu'ici —
# tag GitLab, instantané public GitHub, release GitHub, attente des paquets, vérification.
# Chaque étape s'arrête NET (message + code de sortie) au premier défaut ; rien
# n'est rattrapé en silence.
#
#   outils/sortir-version.sh v0.7.3            # sortie réelle
#   outils/sortir-version.sh v0.7.3 --simule   # trace chaque commande, ne pousse rien
#
# Le jeton GitHub est lu comme dans outils/publier-github.sh : même entrée du coffre
# (~/.config/acvram/jetons.gpg, « github-anticitoyun »), jamais affiché ni journalisé —
# passé à `gh` par variable d'environnement le temps d'un seul appel, jamais exporté
# globalement, jamais dans une commande tracée en clair (--simule ne l'imprime jamais).
set -euo pipefail

DEPOT_GH="anticitoyun/anticitoyen-vram"
NOTES_DIR="docs/notes"

V="${1:-}"
[ -n "$V" ] || { echo "usage : sortir-version.sh vX.Y.Z [--simule]" >&2; exit 64; }
case "$V" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "REFUS : version attendue sous la forme vX.Y.Z (reçu « $V »)" >&2; exit 64 ;;
esac
VNUM="${V#v}"
SIMULE=0
[ "${2:-}" = --simule ] && SIMULE=1

DEPOT="$(git rev-parse --show-toplevel)"
cd "$DEPOT"

# --simule : trace la commande (masquée si elle porte un jeton) au lieu de l'exécuter.
_executer() {
  if [ "$SIMULE" = 1 ]; then
    printf '[simulé] %s\n' "$*"
  else
    "$@"
  fi
}

echo "== 1. main propre, à la version $VNUM"
BRANCHE=$(git branch --show-current)
[ "$BRANCHE" = main ] || { echo "REFUS : branche courante « $BRANCHE », attendu main" >&2; exit 65; }
# Pièce 282 : les fichiers NON SUIVIS sont ignorés ici — un scratchpad/<piece>-<date>/ d'une
# pièce en cours, sur main, est l'état NORMAL du dépôt (scratchpad/ n'est pas gitignoré en
# bloc, REGLES) et n'a rien à voir avec ce qui sera tagué (seul le contenu SUIVI l'est). Le
# script de chef le jour de la 281 tolérait déjà ce cas ; un `git status --porcelain` nu
# aurait refusé une main par ailleurs prête.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "REFUS : main n'est pas propre (fichiers suivis modifiés ou indexés) :" >&2
  git status --short --untracked-files=no >&2
  exit 65
fi
PYVER=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
[ "$PYVER" = "$VNUM" ] || { echo "REFUS : pyproject.toml porte $PYVER, pas $VNUM" >&2; exit 65; }

echo "== 2. notes de release GitHub présentes"
NOTES="$NOTES_DIR/release-$V-github.md"
[ -f "$NOTES" ] || {
  echo "REFUS : $NOTES introuvable — écrire les notes de cette release avant de la sortir" >&2
  exit 66
}

echo "== 3. tag annoté $V, poussé sur GitLab"
if git rev-parse -q --verify "refs/tags/$V" >/dev/null; then
  echo "REFUS : le tag $V existe déjà" >&2
  exit 67
fi
_executer git tag -a "$V" -m "acvram $V"
_executer git push origin "$V"

echo "== 4. instantané public GitHub (publier-github.sh sous carte.sh)"
if [ "$SIMULE" = 1 ]; then
  echo "[simulé] outils/carte.sh outils/publier-github.sh --pousser"
else
  ACVRAM_NOM="sortir-version-$V" outils/carte.sh outils/publier-github.sh --pousser \
    || { echo "REFUS : publier-github.sh a échoué — voir sa sortie ci-dessus" >&2; exit 68; }
fi

echo "== 5. release GitHub $V (notes : $NOTES)"
JETON=""
if [ "$SIMULE" = 1 ]; then
  echo "[simulé] gh release create $V --repo $DEPOT_GH --title \"acvram $V\" --notes-file $NOTES"
else
  JETON=$("$HOME/.config/acvram/jetons-acvram.sh" jeton github-anticitoyun 2>/dev/null) \
    || { echo "REFUS : jeton github-anticitoyun introuvable au coffre" >&2; exit 69; }
  [ -n "$JETON" ] || { echo "REFUS : jeton github-anticitoyun vide au coffre" >&2; exit 69; }
  GH_TOKEN="$JETON" gh release create "$V" --repo "$DEPOT_GH" --title "acvram $V" --notes-file "$NOTES" \
    || { echo "REFUS : gh release create a échoué" >&2; exit 69; }
fi

echo "== 6. attente du run « Release packages » déclenché par la release"
if [ "$SIMULE" = 1 ]; then
  echo "[simulé] gh run list --repo $DEPOT_GH --workflow=\"Release packages\" (recherche du run)"
  echo "[simulé] gh run watch <id> --repo $DEPOT_GH --exit-status"
else
  ID=""
  for _ in $(seq 1 20); do   # ≤ 20 × 15 s = 5 min pour que le run apparaisse ; espacé, jamais une boucle serrée
    ID=$(GH_TOKEN="$JETON" gh run list --repo "$DEPOT_GH" --workflow="Release packages" \
           --json databaseId,event,headBranch -q ".[] | select(.event==\"release\") | .databaseId" 2>/dev/null | head -1)
    [ -n "$ID" ] && break
    sleep 15
  done
  [ -n "$ID" ] || { echo "REFUS : aucun run « Release packages » trouvé pour $V après 5 min" >&2; exit 70; }
  echo "run $ID"
  GH_TOKEN="$JETON" gh run watch "$ID" --repo "$DEPOT_GH" --exit-status \
    || { echo "REFUS : le run $ID a échoué — voir $DEPOT_GH/actions/runs/$ID" >&2; exit 70; }
fi

echo "== 7. vérification de la release publiée"
if [ "$SIMULE" = 1 ]; then
  echo "[simulé] outils/verifier-release.sh $V"
else
  outils/verifier-release.sh "$V" || { echo "REFUS : verifier-release.sh a signalé un défaut" >&2; exit 71; }
fi

echo "$V sortie."
