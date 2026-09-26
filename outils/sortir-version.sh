#!/usr/bin/env bash
# sortir-version.sh vX.Y.Z [--simule] [--depuis N]
#
# Pièce 281 (chef, 26/09) : codifie la sortie de version faite à la main jusqu'ici —
# tag GitLab, instantané public GitHub, release GitHub, attente des paquets, vérification.
# Chaque étape s'arrête NET (message + code de sortie) au premier défaut ; rien
# n'est rattrapé en silence.
#
#   outils/sortir-version.sh v0.7.3               # sortie réelle
#   outils/sortir-version.sh v0.7.3 --simule      # trace chaque commande, ne pousse rien
#   outils/sortir-version.sh v0.7.3 --depuis 5     # reprise après un échec en cours de route
#                                                  # (285, chef : ex. la CI publique a échoué à
#                                                  # l'étape 4 alors que le tag était déjà posé —
#                                                  # une relance sans --depuis tombe sur le code 67).
#                                                  # N ∈ {4,5,6,7} : le tag DOIT déjà exister ET être
#                                                  # ancêtre de main, sinon REFUS (72) — --depuis ne
#                                                  # doit jamais reprendre sur un tag d'une autre sortie.
#
# Le jeton GitHub est lu comme dans outils/publier-github.sh : même entrée du coffre
# (~/.config/acvram/jetons.gpg, « github-anticitoyun »), jamais affiché ni journalisé —
# passé à `gh` par variable d'environnement le temps d'un seul appel, jamais exporté
# globalement, jamais dans une commande tracée en clair (--simule ne l'imprime jamais).
set -euo pipefail

DEPOT_GH="anticitoyun/anticitoyen-vram"
NOTES_DIR="docs/notes"

V="${1:-}"
[ -n "$V" ] || { echo "usage : sortir-version.sh vX.Y.Z [--simule] [--depuis N]" >&2; exit 64; }
case "$V" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "REFUS : version attendue sous la forme vX.Y.Z (reçu « $V »)" >&2; exit 64 ;;
esac
VNUM="${V#v}"
shift
SIMULE=0
DEPUIS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --simule) SIMULE=1; shift ;;
    --depuis) DEPUIS="${2:-}"; shift 2 ;;
    *) echo "REFUS : argument inconnu « $1 »" >&2; exit 64 ;;
  esac
done
case "$DEPUIS" in
  1) ;;
  4|5|6|7) ;;
  *) echo "REFUS : --depuis attend 4, 5, 6 ou 7 (reçu « $DEPUIS »)" >&2; exit 64 ;;
esac

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

if [ "$DEPUIS" -le 3 ]; then
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
    echo "REFUS : le tag $V existe déjà — reprendre avec --depuis si la sortie a échoué après ce point" >&2
    exit 67
  fi
  _executer git tag -a "$V" -m "acvram $V"
  _executer git push origin "$V"
else
  NOTES="$NOTES_DIR/release-$V-github.md"
  echo "== --depuis $DEPUIS : vérification du tag $V avant reprise"
  # 285 (chef) : --depuis ne doit JAMAIS reprendre sur le tag d'une AUTRE sortie — le tag doit
  # exister ET être un ancêtre de main (posé par CETTE branche, pas oublié d'une release passée
  # ou d'un autre dépôt de travail).
  git rev-parse -q --verify "refs/tags/$V" >/dev/null \
    || { echo "REFUS : --depuis $DEPUIS mais le tag $V n'existe pas — rien à reprendre" >&2; exit 72; }
  git merge-base --is-ancestor "refs/tags/$V" main \
    || { echo "REFUS : --depuis $DEPUIS mais le tag $V n'est pas un ancêtre de main" >&2; exit 72; }
  echo "  tag $V trouvé, ancêtre de main — reprise à l'étape $DEPUIS"
fi

if [ "$DEPUIS" -le 4 ]; then
  echo "== 4. instantané public GitHub (publier-github.sh sous carte.sh)"
  if [ "$SIMULE" = 1 ]; then
    echo "[simulé] outils/carte.sh outils/publier-github.sh --pousser"
  else
    ACVRAM_NOM="sortir-version-$V" outils/carte.sh outils/publier-github.sh --pousser \
      || { echo "REFUS : publier-github.sh a échoué — voir sa sortie ci-dessus" >&2; exit 68; }
  fi
fi

JETON=""
if [ "$DEPUIS" -le 5 ]; then
  echo "== 5. release GitHub $V (notes : $NOTES)"
  if [ "$SIMULE" = 1 ]; then
    echo "[simulé] gh release create $V --repo $DEPOT_GH --title \"acvram $V\" --notes-file $NOTES"
  else
    JETON=$("$HOME/.config/acvram/jetons-acvram.sh" jeton github-anticitoyun 2>/dev/null) \
      || { echo "REFUS : jeton github-anticitoyun introuvable au coffre" >&2; exit 69; }
    [ -n "$JETON" ] || { echo "REFUS : jeton github-anticitoyun vide au coffre" >&2; exit 69; }
    GH_TOKEN="$JETON" gh release create "$V" --repo "$DEPOT_GH" --title "acvram $V" --notes-file "$NOTES" \
      || { echo "REFUS : gh release create a échoué" >&2; exit 69; }
  fi
fi

if [ "$DEPUIS" -le 6 ]; then
  echo "== 6. attente du run « Release packages » déclenché par la release"
  if [ "$SIMULE" = 1 ]; then
    echo "[simulé] gh run list --repo $DEPOT_GH --workflow=\"Release packages\" (recherche du run, headBranch=$V)"
    echo "[simulé] gh run watch <id> --repo $DEPOT_GH --exit-status"
  else
    if [ -z "$JETON" ]; then
      JETON=$("$HOME/.config/acvram/jetons-acvram.sh" jeton github-anticitoyun 2>/dev/null) \
        || { echo "REFUS : jeton github-anticitoyun introuvable au coffre" >&2; exit 69; }
    fi
    ID=""
    for _ in $(seq 1 20); do   # ≤ 20 × 15 s = 5 min pour que le run apparaisse ; espacé, jamais une boucle serrée
      # 285 (chef) : un `head -1` sans filtrer sur headBranch peut prendre le run d'une AUTRE
      # sortie (release-triggered workflow : head_branch = le TAG, pas main) — filtré sur $V.
      ID=$(GH_TOKEN="$JETON" gh run list --repo "$DEPOT_GH" --workflow="Release packages" \
             --json databaseId,event,headBranch -q ".[] | select(.event==\"release\" and .headBranch==\"$V\") | .databaseId" 2>/dev/null | head -1)
      [ -n "$ID" ] && break
      sleep 15
    done
    [ -n "$ID" ] || { echo "REFUS : aucun run « Release packages » trouvé pour $V après 5 min" >&2; exit 70; }
    echo "run $ID"
    GH_TOKEN="$JETON" gh run watch "$ID" --repo "$DEPOT_GH" --exit-status \
      || { echo "REFUS : le run $ID a échoué — voir $DEPOT_GH/actions/runs/$ID" >&2; exit 70; }
  fi
fi

if [ "$DEPUIS" -le 7 ]; then
  echo "== 7. vérification de la release publiée"
  if [ "$SIMULE" = 1 ]; then
    echo "[simulé] outils/verifier-release.sh $V"
  else
    outils/verifier-release.sh "$V" || { echo "REFUS : verifier-release.sh a signalé un défaut" >&2; exit 71; }
  fi
fi

echo "$V sortie."
