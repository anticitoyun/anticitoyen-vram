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
           "$PKG/usr/share/doc/acvram" \
           "$PKG/usr/share/applications" \
           "$PKG/usr/share/icons/hicolor/scalable/apps"
cp -r acvram pyproject.toml install.sh README.md LICENSE "$PKG/usr/share/acvram/"
# Lot poste 20/09 : outils de vérification du poste (documentation, aucun chemin de machine imposé).
install -d "$PKG/usr/share/acvram/poste" && cp -r outils/poste/. "$PKG/usr/share/acvram/poste/"
# Le chemin des modèles du poste de développement ne part pas dans le paquet : gabarit à remplir.
sed -i "s#=/mnt/[^ ]*models_acvram#=<racine des modeles convertis>#" "$PKG/usr/share/acvram/poste/environment.d/"*.conf
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

# ---- shebang absolu sur les lanceurs Python --------------------------------
# `#!/usr/bin/env python3` suit le PATH de qui lance le programme : un
# linuxbrew/pyenv/conda avant /usr/bin fait rater python3-gi (installe au
# niveau systeme, jamais dans ces environnements alternatifs) — crash muet au
# premier clic sur l'icone. Seul /usr/bin/python3 est garanti avoir les
# dependances systeme listees dans DEBIAN/control (Depends: python3-gi, ...).
if grep -lE '^#!.*env python3' packaging/* 2>/dev/null; then
    echo "REFUS : un lanceur packaging/ utilise 'env python3' au lieu du chemin absolu /usr/bin/python3." >&2
    exit 1
fi

# ---- sous-commandes sous les deux interpretes ------------------------------
# --help construit tout le parseur (choix, aide, defauts) : un texte d'aide
# malforme (ex : un '%' litteral interprete comme format par argparse) casse
# le CLI entier des le premier appel, jamais vu par la suite de tests qui
# n'invoque jamais --help. Teste sous .venv (dev) ET /usr/bin/python3 (ce que
# le paquet execute reellement en dernier ressort) pour attraper une
# dependance a une bibliotheque absente d'un des deux.
for interp in .venv/bin/python /usr/bin/python3; do
    [ -x "$interp" ] || continue
    for sub in serve convert doctor; do
        if ! PYTHONPATH="$PKG/usr/share/acvram" "$interp" -c "
import sys; sys.argv=['acvram','$sub','--help']
from acvram.cli import main
try:
    main()
except SystemExit as e:
    sys.exit(0 if e.code in (0, None) else 1)
" >/dev/null 2>&1; then
            echo "REFUS : 'acvram $sub --help' echoue sous $interp." >&2
            exit 1
        fi
    done
done

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
            pip install --quiet --only-binary=:all: 'cuda-toolkit[nvcc,cccl]==13.0.*' || true
        else
            INDEX="https://download.pytorch.org/whl/cu124"
        fi
    fi
    pip install --quiet --index-url "$INDEX" torch
    # 0.6.33 (poste7 20/09 14 h 10) : la tour de vision (Gemma 4, Qwen3-VL) passe par les classes
    # transformers, version EPINGLEE = celle qui charge gemma4 sur le poste (5.17.0) ; import
    # paresseux dans acvram/engine/vision.py : un alias texte ne l importe jamais.
    pip install --quiet "transformers==5.17.0" pillow
    # processing_gemma4 importe torchvision : la version accordée à torch (même index cu*), jamais une autre
    pip install --quiet --index-url "$INDEX" torchvision
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

# ---- bureau : entree de menu, icone, lanceur de console --------------------
# Le paquet s'installait sans rien de visible. Un utilisateur qui vient de
# faire `dpkg -i` n'a aucun moyen de savoir que quelque chose est installe :
# pas d'icone, pas d'entree de menu, et un serveur qui ne repond qu'a des
# clients OpenAI donc muet tant qu'on ne le sollicite pas.
#
# L'entree de menu n'appelle PAS `acvram serve` : celui-ci exige un repertoire
# de modele et il n'y a pas de defaut raisonnable. Elle appelle un lanceur qui
# cherche un serveur deja en marche et ouvre sa console ; s'il n'en trouve
# aucun, il dit quoi taper au lieu de deviner un modele.
install -m 755 packaging/acvram-console "$PKG/usr/bin/acvram-console"
# La fenetre native (GTK + WebKit) : meme page, meme theme, sans navigateur,
# avec le choix du modele, le demarrage du serveur, le plein ecran, la galerie
# dans sa fenetre, et une barre d'etat qui lit les capteurs. Elle a besoin de
# carte.sh pour prendre le verrou de la carte qu'elle sert.
install -m 755 packaging/acvram-gui "$PKG/usr/bin/acvram-gui"
install -d "$PKG/usr/share/acvram/langues"
install -m 644 packaging/langues/*.json "$PKG/usr/share/acvram/langues/"   # traductions de la GUI
install -m 644 packaging/logo-acvram.jpg "$PKG/usr/share/acvram/logo-acvram.jpg"   # logo de la GUI (barre, bandeau, fond)
install -d "$PKG/usr/share/acvram/galerie" && install -m 644 packaging/galerie/*.jpg "$PKG/usr/share/acvram/galerie/"   # galerie de la section À propos
install -D -m 755 outils/carte.sh "$PKG/usr/share/acvram/carte.sh"
install -m 644 packaging/acvram.desktop "$PKG/usr/share/applications/acvram.desktop"
install -m 644 packaging/acvram.svg \
        "$PKG/usr/share/icons/hicolor/scalable/apps/acvram.svg"

# ---- métadonnées -----------------------------------------------------------
cat > "$PKG/DEBIAN/control" <<CTRL
Package: acvram
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), python3-venv, python3-pip, ca-certificates, curl, python3-gi, gir1.2-gtk-3.0, gir1.2-webkit2-4.1, python3-psutil
Recommends: lm-sensors, nvidia-driver-575 | nvidia-driver-580 | nvidia-driver-595
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
 .
 Soutenir : https://buymeacoffee.com/anticitoyen
CTRL

# Éco d'horloge par défaut (poste7-eco-2700-defaut-19-09, décision utilisateur
# 19/09) : le service pose `sudo -n nvidia-smi -i N -lgc 2700,2700` et rend
# `-rgc`. Droit RESTREINT à ces deux formes, pour les membres du groupe sudo
# (le paquet ne connaît pas l'utilisateur) ; conffile, retiré à la purge.
install -d -m 755 "$PKG/etc/sudoers.d"
cat > "$PKG/etc/sudoers.d/acvram-nvidia-smi" <<'SUDOERS'
# acvram : verrou d'horloge du service (acvram eco 2700|2100|off), rien d'autre
# Formes EXACTES (ce sudo refuse les jokers dans les arguments, visudo -c) :
# cartes 0 et 1, modes 2700 / 2100, relâchement.
%sudo ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -i 0 -lgc 2700\,2700, /usr/bin/nvidia-smi -i 0 -lgc 2100\,2100, /usr/bin/nvidia-smi -i 0 -rgc, /usr/bin/nvidia-smi -i 1 -lgc 2700\,2700, /usr/bin/nvidia-smi -i 1 -lgc 2100\,2100, /usr/bin/nvidia-smi -i 1 -rgc
SUDOERS
chmod 440 "$PKG/etc/sudoers.d/acvram-nvidia-smi"
visudo -cf "$PKG/etc/sudoers.d/acvram-nvidia-smi" >/dev/null || { echo "sudoers invalide" >&2; exit 1; }
echo "/etc/sudoers.d/acvram-nvidia-smi" > "$PKG/DEBIAN/conffiles"

cat > "$PKG/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
set -e
# Sans ces deux rafraichissements, l'entree de menu et l'icone n'apparaissent
# qu'a la prochaine ouverture de session : l'utilisateur conclut a tort que le
# paquet n'a rien installe.
[ -x /usr/bin/update-desktop-database ] && update-desktop-database -q /usr/share/applications || true
[ -x /usr/bin/gtk-update-icon-cache ] && gtk-update-icon-cache -qf /usr/share/icons/hicolor || true
exit 0
POSTINST
chmod 755 "$PKG/DEBIAN/postinst"

dpkg-deb --build --root-owner-group "$PKG" >/dev/null
mv "$PKG.deb" .
echo "construit : $(basename "$PKG").deb ($(du -h "$(basename "$PKG").deb" | cut -f1))"
