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
# LISTE BLANCHE, jamais une liste noire. La liste noire precedente
# (FEUILLE-DE-ROUTE, REPRISE, CHANTIER-*) laissait passer les documents de
# TRAVAIL : PROTOCOLES-EN-ATTENTE.md porte trois noms de sessions internes,
# PREDICTION-CAMPAGNE-9SEPT.md en porte un, FUSIONS-LIBRES-PARC.md cite le
# chemin du parc de modeles. Rien de secret, mais rien qui concerne qui
# installe le paquet — et une liste noire oublie toujours le document ecrit
# apres elle.
#
# N'ajouter ici qu'un document destine a L'UTILISATEUR du paquet, pas a nous.
DOCS_PUBLIQUES="
ARCHITECTURE.md
BRANCHER-UN-CLIENT.md
MATERIEL.md
FORMAT-3BITS.md
PROTOCOLE-ENERGIE.md
PROTOCOLE-PERPLEXITE.md
REFERENCES.md
BIBLIOGRAPHIE.md
"
for nom in $DOCS_PUBLIQUES; do
    [ -f "docs/$nom" ] || continue
    install -m 644 "docs/$nom" "$PKG/usr/share/doc/acvram/"
done
find "$PKG" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# ---- controle de contenu ---------------------------------------------------
# La liste blanche choisit les FICHIERS ; elle ne dit rien de ce qu'ils
# CONTIENNENT. Le premier paquet construit apres elle portait encore deux noms
# de sessions de travail internes, dans des commentaires de `layers.py` et de
# `model.py` — deux fichiers de code que personne n'aurait pense a relire pour
# cela. Un motif trouve arrete la construction : mieux vaut ne pas livrer que
# livrer ce qu'on n'a pas relu.
MOTIFS_INTERDITS='anticitoyenlm|/home/[a-z]+/Bureau|9c9efa0|234ead47'
if trouve=$(grep -rlniE "$MOTIFS_INTERDITS" "$PKG" 2>/dev/null); then
    echo "REFUS : le paquet contient un motif interdit." >&2
    grep -rniE "$MOTIFS_INTERDITS" "$PKG" 2>/dev/null | sed "s|$PKG||" | head -20 >&2
    exit 1
fi

# ---- lanceur ---------------------------------------------------------------
cat > "$PKG/usr/bin/acvram" <<'LANCEUR'
#!/usr/bin/env bash
# Lanceur acvram : amorce l'environnement au premier appel, puis s'efface.
set -euo pipefail
BASE="${ACVRAM_HOME:-$HOME/.local/share/acvram}"
VENV="$BASE/venv"
SRC="/usr/share/acvram"

# La version qui compte est celle du venv, pas celle du paquet : le lanceur
# ne verifiait QUE l'existence du venv, donc une mise a jour du .deb n'avait
# aucun effet — dpkg annoncait 0.3.0 pendant que `acvram --version` rendait
# 0.2.0. Un defaut muet : rien ne casse, la mise a jour ne fait simplement rien.
VERSION_SRC=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$SRC/acvram/__init__.py" 2>/dev/null || true)
# -I isole l interpreteur : sans lui, un `acvram` lance depuis un arbre de
# developpement importe le paquet du REPERTOIRE COURANT et non celui du venv,
# les deux versions paraissent egales, et la mise a jour ne se declenche pas.
VERSION_VENV=$("$VENV/bin/python" -I -c 'import acvram;print(acvram.__version__)' 2>/dev/null || true)

if [ -x "$VENV/bin/acvram" ] && [ -n "$VERSION_SRC" ] \
   && [ "$VERSION_SRC" != "$VERSION_VENV" ]; then
    echo "acvram : $VERSION_VENV installe, $VERSION_SRC disponible — mise a jour"
    # torch n'est pas retelecharge : seul le paquet acvram est reinstalle.
    COPIE="$BASE/src"
    rm -rf "$COPIE"
    cp -r "$SRC" "$COPIE"
    "$VENV/bin/pip" install --quiet --no-deps --force-reinstall "$COPIE"
    rm -rf "$COPIE"
    echo "acvram : a jour en $VERSION_SRC."
fi

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
Maintainer: Anticitoyen <anticitoyen@users.noreply.gitlab.com>
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
