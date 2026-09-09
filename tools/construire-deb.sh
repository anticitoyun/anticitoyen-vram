#!/usr/bin/env bash
# Construit acvram_<version>_amd64.deb.
#
# Le paquet ne peut pas embarquer torch (plus de trois gigaoctets, choisi selon
# le GPU présent) : il installe le code sous /usr/share/acvram et un lanceur
# /usr/bin/acvram qui amorce l'environnement au premier appel — vérifiable avec
# `acvram doctor`, supprimable avec `rm -rf ~/.local/share/acvram`.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import re;print(re.search(r'__version__ = \"([^\"]+)\"', open('acvram/__init__.py').read()).group(1))")
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/acvram_${VERSION}_amd64"

# ---- code ------------------------------------------------------------------
install -d "$PKG/usr/share/acvram" "$PKG/usr/bin" "$PKG/DEBIAN" \
           "$PKG/usr/share/doc/acvram"
cp -r acvram pyproject.toml install.sh README.md LICENSE "$PKG/usr/share/acvram/"
# docs/ contenait 8 fichiers de mesure INTERNE — comparatifs, rebancs,
# releves de repetabilite — qui n'ont rien a faire dans un paquet distribue,
# et FEUILLE-DE-ROUTE.md y porte le chemin et le nom d'utilisateur de la
# machine de developpement. Seuls les documents utiles a qui installe sont
# copies, et jamais un .tsv ni un .txt de mesure.
for d in docs/*.md; do
    case "$(basename "$d")" in
        FEUILLE-DE-ROUTE.md|REPRISE.md|CHANTIER-*) continue ;;
    esac
    install -m 644 "$d" "$PKG/usr/share/doc/acvram/"
done
find "$PKG" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# ---- lanceur ---------------------------------------------------------------
cat > "$PKG/usr/bin/acvram" <<'LANCEUR'
#!/usr/bin/env bash
# Lanceur acvram : amorce l'environnement au premier appel, puis s'efface.
set -euo pipefail
BASE="${ACVRAM_HOME:-$HOME/.local/share/acvram}"
VENV="$BASE/venv"
SRC="/usr/share/acvram"

if [ ! -x "$VENV/bin/acvram" ]; then
    echo "acvram : premier lancement, préparation de l'environnement dans $VENV"
    echo "         (torch se choisit selon les GPU présents ; plusieurs Gio)"
    mkdir -p "$BASE"
    python3 -m venv "$VENV"
    # shellcheck disable=SC1091
    . "$VENV/bin/activate"
    pip install --quiet --upgrade pip wheel
    INDEX="https://download.pytorch.org/whl/cpu"
    if command -v nvidia-smi >/dev/null 2>&1; then
        CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
        if echo "$CAPS" | grep -qE '^(1[0-9])\.'; then
            INDEX="https://download.pytorch.org/whl/cu130"
            pip install --quiet --only-binary=:all: 'cuda-toolkit[nvcc]' || true
        else
            INDEX="https://download.pytorch.org/whl/cu124"
        fi
    fi
    pip install --quiet --index-url "$INDEX" torch
    # /usr/share est en lecture seule : construire depuis une copie, sinon
    # setuptools échoue en voulant y écrire acvram.egg-info
    COPIE="$BASE/src"
    rm -rf "$COPIE"
    cp -r "$SRC" "$COPIE"
    pip install --quiet "$COPIE"
    rm -rf "$COPIE"
    echo "acvram : prêt. « acvram doctor » pour vérifier."
fi
exec "$VENV/bin/acvram" "$@"
LANCEUR
chmod 755 "$PKG/usr/bin/acvram"

# ---- métadonnées -----------------------------------------------------------
cat > "$PKG/DEBIAN/control" <<CTRL
Package: acvram
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), python3-venv, python3-pip, ca-certificates
Recommends: nvidia-driver-575 | nvidia-driver-580 | nvidia-driver-595
Maintainer: Anticitoyen <anticitoyen@users.noreply.github.com>
Homepage: https://outils.nuages.noho.st/gitlab/anticitoyen/anticitoyen-vram
Description: serveur d'inférence LLM pour GPU hétérogènes (NVFP4 + INT4)
 Serveur d'inférence compatible OpenAI qui donne à chaque GPU le format de
 quantification que son silicium sait lire (NVFP4 sur Blackwell, INT4 sur
 Ampere) et traite la mémoire comme une hiérarchie VRAM-VRAM-RAM mesurée.
 Convertit les points de contrôle safetensors, GGUF et EXL3.
 .
 L'environnement Python (torch inclus) s'amorce au premier lancement dans
 ~/.local/share/acvram ; le paquet lui-même reste léger.
CTRL

dpkg-deb --build --root-owner-group "$PKG" >/dev/null
mv "$PKG.deb" .
echo "construit : $(basename "$PKG").deb ($(du -h "$(basename "$PKG").deb" | cut -f1))"
