#!/usr/bin/env bash
# Construit acvram-parc_<v>_amd64.deb à partir de parc/. Usage : tools/construire-deb-parc.sh [--stage-only DIR]
# Refus si l'arbre du paquet contient un chemin de machine (sage-deb-parc-portable-20-09 § 4).
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(tr -d '[:space:]' < parc/VERSION)
if [ "${1:-}" = "--stage-only" ]; then STAGE=${2:?dossier}; mkdir -p "$STAGE"; else STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT; fi
PKG="$STAGE/acvram-parc_${VERSION}_amd64"
rm -rf "$PKG"
install -d "$PKG/DEBIAN" "$PKG/usr/bin" "$PKG/usr/share/acvram-parc/lib" "$PKG/usr/share/acvram-parc/gabarits" \
           "$PKG/usr/share/acvram-parc/systemd-user" "$PKG/usr/share/applications" \
           "$PKG/usr/share/icons/hicolor/scalable/apps" "$PKG/usr/share/icons/hicolor/256x256/apps" "$PKG/usr/share/doc/acvram-parc"
for s in claude-modele kimi-modele claude-modeles kimi-modeles modeles-a-jour integrite-modeles telecharger-modele parc-installer; do
    install -m 755 "parc/bin/$s" "$PKG/usr/bin/$s"
done
install -m 644 parc/lib/acvram_parc.py "$PKG/usr/share/acvram-parc/lib/"
install -d "$PKG/usr/share/acvram-parc/lib/menu_modeles"
for f in parc/lib/menu_modeles/*.py; do install -m 644 "$f" "$PKG/usr/share/acvram-parc/lib/menu_modeles/"; done
install -m 644 parc/share/kimi-menu.lib.sh "$PKG/usr/share/acvram-parc/"
install -m 644 parc/share/gabarits/* "$PKG/usr/share/acvram-parc/gabarits/"
install -m 644 parc/share/systemd-user/* "$PKG/usr/share/acvram-parc/systemd-user/"
[ -f parc/share/icones/claude-modeles.svg ] && install -m 644 parc/share/icones/claude-modeles.svg "$PKG/usr/share/icons/hicolor/scalable/apps/"
[ -f parc/share/icones/kimi-modele.png ] && install -m 644 parc/share/icones/kimi-modele.png "$PKG/usr/share/icons/hicolor/256x256/apps/"
for d in parc/share/desktop/*.desktop; do install -m 644 "$d" "$PKG/usr/share/applications/$(basename "$d")"; done
install -m 644 parc/GUIDE.md "$PKG/usr/share/doc/acvram-parc/README.md" 2>/dev/null || true
sed "s/@VERSION@/$VERSION/" parc/DEBIAN/control > "$PKG/DEBIAN/control"
find "$PKG" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
MOTIFS='/mnt/|/opt/ia|/home/[a-z]+|/media/'   # chemins seulement : le nom d'auteur n'en est pas un (Sage 20/09)
if trouve=$(grep -rnE "$MOTIFS" "$PKG/usr" 2>/dev/null); then
    echo "REFUS : chemin de machine dans le paquet :" >&2; echo "$trouve" | sed "s|$PKG||" | head -20 >&2; exit 1
fi
if grep -lE '^#!.*env python3' "$PKG/usr/bin/"* 2>/dev/null; then echo "REFUS : shebang env python3 dans /usr/bin" >&2; exit 1; fi
for s in claude-modeles kimi-modeles modeles-a-jour integrite-modeles parc-installer; do python3 -m py_compile "$PKG/usr/bin/$s"; done
for s in claude-modele kimi-modele telecharger-modele; do bash -n "$PKG/usr/bin/$s"; done
find "$PKG" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
if [ "${1:-}" = "--stage-only" ]; then echo "$PKG"; exit 0; fi
dpkg-deb --root-owner-group --build "$PKG" "acvram-parc_${VERSION}_amd64.deb" >/dev/null
echo "acvram-parc_${VERSION}_amd64.deb"
