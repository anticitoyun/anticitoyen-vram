#!/bin/bash
# Pièce 259 : vérifier une release GitHub d'acvram APRÈS sa publication — ce que release.yml annonce est là, sous
# le nom attendu, et chaque paquet s'ouvre. Rien en sudo, rien sur l'installation système : le Flatpak s'installe
# dans une installation utilisateur SÉPARÉE (FLATPAK_USER_DIR dédié) et `acvram doctor` tourne dans son bac à
# sable — c'est ce qui lève la réserve de la 241 (le .so précompilé par la CI charge-t-il sur une carte réelle ?).
#
#     outils/verifier-release.sh v0.7.0                       # télécharge (gh) dans ~/.cache/acvram/releases/v0.7.0
#     outils/verifier-release.sh v0.7.0 --dossier /chemin      # ailleurs (jamais /tmp : un rejeu doit retrouver les fichiers)
#     outils/verifier-release.sh v0.7.0 --simule /dossier      # release simulée : pas de gh, les fichiers sont déjà là
#     ... --sans-flatpak                                       # saute le bras Flatpak (installation ET doctor)
#     ... --flatpak-installer                                   # bras Flatpak : installation seule (≈ 3 Go tirés par extra-data), SANS carte
#     ... --flatpak-doctor                                      # bras Flatpak : doctor seul, installation déjà faite — sous carte.sh
#
# Sortie : une ligne par contrôle, `OK` / `MANQUE` / `FAUX` / `SAUTÉ`, puis `VERDICT: TENU` (code 0) ou `FAUX` (1) —
# sauf un asset cité dans le corps de la release mais absent des assets joints (pièce 285), qui refuse à part,
# code 65 : un lien mort dans les notes publiées (v0.7.3/acvram-0.7.3.flatpakref, job flatpak échoué, jamais vu)
# ne doit jamais se confondre avec un MANQUE générique dans un script qui surveille CE script. Code 66 : le
# Flatpak installé n'est pas la version $V après attente bornée (v0.7.4, Pages n'avait pas encore servi le
# nouveau dépôt OSTree — l'ancienne 0.7.2 s'installait sans erreur et le contrôle disait OK).
# Ce que ce script NE prouve PAS : que le .deb s'installe (dpkg-deb --info/--contents seulement, pas d'installation),
# ni que les RPM se construisent (rpm absent sur ce poste : nom et taille seulement).
set -uo pipefail

TAG=""; DOSSIER=""; SIMULE=""; FLATPAK=1
DEPOT=${ACVRAM_DEPOT_GITHUB:-anticitoyun/anticitoyen-vram}
APP=io.github.anticitoyen.acvram
while [ $# -gt 0 ]; do
  case "$1" in
    --dossier) DOSSIER=$2; shift 2 ;;
    --simule) SIMULE=$2; shift 2 ;;
    --sans-flatpak) FLATPAK=0; shift ;;
    --flatpak-installer) FLATPAK=installer; shift ;;
    --flatpak-doctor) FLATPAK=doctor; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    v*) TAG=$1; shift ;;
    *) echo "argument inconnu : $1" >&2; exit 64 ;;
  esac
done
[ -n "$TAG" ] || { echo "usage : verifier-release.sh vX.Y.Z [--dossier D | --simule D] [--sans-flatpak]" >&2; exit 64; }
V=${TAG#v}
if [ -n "$SIMULE" ]; then DOSSIER=$SIMULE; else DOSSIER=${DOSSIER:-$HOME/.cache/acvram/releases/$TAG}; fi
case "$DOSSIER" in /tmp/*) echo "refus : $DOSSIER est sous /tmp (un rejeu doit retrouver les fichiers)" >&2; exit 64 ;; esac
mkdir -p "$DOSSIER"

FAUX=0
ok()     { printf 'OK      %s\n' "$*"; }
manque() { printf 'MANQUE  %s\n' "$*"; FAUX=1; }
faux()   { printf 'FAUX    %s\n' "$*"; FAUX=1; }
saute()  { printf 'SAUTÉ   %s\n' "$*"; }

# ---- 1. téléchargement : gh s'il est authentifié, sinon l'API publique (dépôt public, aucun jeton nécessaire) ------
if [ -z "$SIMULE" ]; then
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release view "$TAG" --repo "$DEPOT" --json tagName,isDraft,assets \
       --jq '"release " + .tagName + (if .isDraft then " (BROUILLON)" else "" end) + " : " + (.assets | length | tostring) + " fichiers (gh)"' \
       || { faux "release $TAG introuvable sur $DEPOT"; echo "VERDICT: FAUX"; exit 1; }
    gh release download "$TAG" --repo "$DEPOT" --dir "$DOSSIER" --clobber || faux "gh release download"
    gh release view "$TAG" --repo "$DEPOT" --json assets --jq '.assets[].name' > "$DOSSIER/liste-release.txt"
    gh release view "$TAG" --repo "$DEPOT" --json body --jq '.body' > "$DOSSIER/corps-release.txt"
  else
    # sans jeton : api.github.com (60 requêtes/h) et browser_download_url — un jeton ne transite jamais par ici
    python3 - "$DEPOT" "$TAG" "$DOSSIER" <<'PY' || { faux "release $TAG introuvable sur $DEPOT (API publique)"; echo "VERDICT: FAUX"; exit 1; }
import json, pathlib, sys, urllib.request
depot, tag, dossier = sys.argv[1:4]
with urllib.request.urlopen(f"https://api.github.com/repos/{depot}/releases/tags/{tag}", timeout=30) as r:
    d = json.load(r)
print(f"release {d['tag_name']}{' (BROUILLON)' if d.get('draft') else ''} : {len(d['assets'])} fichiers (API publique)")
(pathlib.Path(dossier) / "liste-release.txt").write_text("".join(a["name"] + "\n" for a in d["assets"]), encoding="utf-8")
(pathlib.Path(dossier) / "corps-release.txt").write_text(d.get("body") or "", encoding="utf-8")
for a in d["assets"]:
    cible = pathlib.Path(dossier) / a["name"]
    with urllib.request.urlopen(a["browser_download_url"], timeout=600) as r, open(cible, "wb") as f:
        while True:
            bloc = r.read(1 << 20)
            if not bloc:
                break
            f.write(bloc)
    print(f"  {a['name']} ({cible.stat().st_size} o)")
PY
  fi
fi
echo "dossier : $DOSSIER"

# ---- 1 bis. chaque asset CITÉ dans le corps de la release existe parmi les assets JOINTS ---------------------------
# Pièce 285 (chef, 26/09) : les notes v0.7.3 citaient .../v0.7.3/acvram-0.7.3.flatpakref alors que ce fichier
# n'a jamais existé (le job flatpak avait échoué, 266 m) — rien ne l'a signalé, la release est restée TENU aux
# yeux de ce script. `corps-release.txt` (téléchargé ci-dessus ; ou fourni tel quel sous --simule, « le dossier
# fait foi ») porte le texte publié ; `liste-release.txt` (ou, en --simule sans elle, les fichiers du dossier)
# porte ce qui est réellement joint. Code de sortie DISTINCT (65) : ce défaut précis ne doit jamais se confondre
# avec un simple MANQUE générique (1) dans un script qui surveille CE script.
if [ -s "$DOSSIER/corps-release.txt" ]; then
  if [ -s "$DOSSIER/liste-release.txt" ]; then
    ASSETS_JOINTS="$DOSSIER/liste-release.txt"
  else
    ASSETS_JOINTS="$DOSSIER/.assets-disque.txt"
    (cd "$DOSSIER" && for f in *; do
       [ -f "$f" ] || continue
       case "$f" in corps-release.txt|liste-release.txt|.assets-disque.txt) continue ;; esac
       printf '%s\n' "$f"
     done) > "$ASSETS_JOINTS"
  fi
  CITES=$(grep -oE "releases/download/$TAG/[^)\"'[:space:]]+" "$DOSSIER/corps-release.txt" | sed 's#.*/##' | sort -u)
  MANQUANTS=""
  while IFS= read -r nom; do
    [ -n "$nom" ] || continue
    grep -qxF "$nom" "$ASSETS_JOINTS" || MANQUANTS="$MANQUANTS $nom"
  done <<< "$CITES"
  if [ -n "$MANQUANTS" ]; then
    faux "corps de release : cité mais absent des assets joints :$MANQUANTS"
    echo "VERDICT: FAUX ($TAG, $DOSSIER)"
    exit 65
  fi
  ok "corps de release : tous les assets cités sont joints"
else
  saute "corps de release : corps-release.txt absent, rien à comparer"
fi
# 259 d : le dossier est un cache — un fichier d'un téléchargement PRÉCÉDENT (l'ancien bundle de la v0.7.0, un .deb
# refait) y reste et se ferait juger comme joint. Ce qui est joint, c'est liste-release.txt ; le reste va dans
# hors-release/ (déplacé, jamais effacé). En --simule, le dossier fait foi.
if [ -z "$SIMULE" ] && [ -s "$DOSSIER/liste-release.txt" ]; then
  for f in "$DOSSIER"/*; do
    [ -f "$f" ] || continue
    case "$(basename "$f")" in liste-release.txt|*.log|*.txt) continue ;; esac
    if ! grep -qxF "$(basename "$f")" "$DOSSIER/liste-release.txt"; then
      mkdir -p "$DOSSIER/hors-release" && mv -f "$f" "$DOSSIER/hors-release/" && echo "HORS    $(basename "$f") : sur disque, pas dans la release — déplacé dans hors-release/"
    fi
  done
fi

# ---- 2. présence et nom de chaque fichier annoncé par release.yml (un seul de chaque) ---------------------------
# job deb → acvram_<V>_amd64.deb ; rpm → acvram-<V>-*.noarch.rpm + acvram-<V>-*.src.rpm ; aur → aur-<V>.tar.gz ;
# flatpak → acvram-<V>.flatpakref (266 i : dépôt OSTree gh-pages, plus de bundle) ; translations → translations-<V>.zip
TROUVE=""
un_seul() {   # un_seul <libellé> <motif glob> → TROUVE = le chemin (vide si absent ou ambigu), une ligne OK/MANQUE/FAUX
  local lib=$1 motif=$2; local -a f=( $DOSSIER/$motif ); TROUVE=""
  if [ ${#f[@]} -eq 1 ] && [ -e "${f[0]}" ]; then ok "$lib : $(basename "${f[0]}") ($(stat -c %s "${f[0]}") o)"; TROUVE=${f[0]}
  elif [ ${#f[@]} -gt 1 ] && [ -e "${f[0]}" ]; then faux "$lib : ${#f[@]} fichiers pour $motif (un seul attendu)"
  else manque "$lib : $motif"; fi
}
un_seul "deb" "acvram_${V}_amd64.deb";          DEB=$TROUVE
un_seul "rpm" "acvram-${V}-*.noarch.rpm";       RPM=$TROUVE
un_seul "src.rpm" "acvram-${V}-*.src.rpm";      SRPM=$TROUVE
un_seul "aur" "aur-${V}.tar.gz";                AUR=$TROUVE
un_seul "flatpak" "acvram-${V}.flatpakref";     FLAT=$TROUVE
{ [ -n "$SIMULE" ] && [ -e "$DOSSIER/acvram-${V}.flatpak" ] || { [ -s "$DOSSIER/liste-release.txt" ] && grep -qxF "acvram-${V}.flatpak" "$DOSSIER/liste-release.txt"; }; } && faux "flatpak : l'ancien bundle acvram-${V}.flatpak est encore joint (266 i : à retirer, > 2 Gio et sans extra-data)"
un_seul "translations" "translations-${V}.zip"; TRAD=$TROUVE

# ---- 3. sommes publiées (si release.yml en produit) ; sinon les sommes calculées, pour le registre ---------------
shopt -s nullglob
SOMMES=( "$DOSSIER"/*.sha256 "$DOSSIER"/SHA256SUMS* "$DOSSIER"/sha256sums* )
shopt -u nullglob
if [ ${#SOMMES[@]} -gt 0 ]; then
  for s in "${SOMMES[@]}"; do
    (cd "$DOSSIER" && sha256sum -c --quiet "$(basename "$s")") && ok "sha256 : $(basename "$s") vérifié" || faux "sha256 : $(basename "$s") ne correspond pas"
  done
else
  saute "sha256 : aucune somme publiée par release.yml — sommes calculées :"
  (cd "$DOSSIER" && for f in *; do [ -f "$f" ] && [ "$f" != "verification.txt" ] && sha256sum "$f"; done) | sed 's/^/        /'
fi

# ---- 4. .deb : en-tête et contenu, SANS installation ----------------------------------------------------------
if [ -n "$DEB" ]; then
  if command -v dpkg-deb >/dev/null; then
    INFO=$(dpkg-deb --info "$DEB" 2>&1) || faux "deb : dpkg-deb --info échoue"
    PKG=$(printf '%s\n' "$INFO" | sed -n 's/^ *Package: *//p'); VER=$(printf '%s\n' "$INFO" | sed -n 's/^ *Version: *//p')
    [ "$PKG" = "acvram" ] && ok "deb : Package acvram" || faux "deb : Package '$PKG' (attendu acvram)"
    case "$VER" in "$V"|"$V-"*) ok "deb : Version $VER" ;; *) faux "deb : Version '$VER' (attendu $V)" ;; esac
    N=$(dpkg-deb --contents "$DEB" 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] && ok "deb : $N entrées (dpkg-deb --contents)" || faux "deb : contenu vide ou illisible"
    # 266 b : pas de `grep -q` en aval d'une commande longue sous pipefail — grep -q ferme le tube au premier
    # succès, dpkg-deb meurt de SIGPIPE et le tube rend faux (« usr/bin/acvram absent » sur la v0.7.0, 274 entrées)
    if [ "$(dpkg-deb --contents "$DEB" 2>/dev/null | grep -c 'usr/bin/acvram')" -gt 0 ]; then ok "deb : usr/bin/acvram présent"; else faux "deb : usr/bin/acvram absent"; fi
  else saute "deb : dpkg-deb absent"; fi
fi

# ---- 5. AUR : PKGBUILD + .SRCINFO, version et somme renseignées ----------------------------------------------
if [ -n "$AUR" ]; then
  L=$(tar tzf "$AUR" 2>/dev/null)
  if [ "$(printf '%s\n' "$L" | grep -cx 'acvram/PKGBUILD')" -gt 0 ] && [ "$(printf '%s\n' "$L" | grep -cx 'acvram/.SRCINFO')" -gt 0 ]; then
    ok "aur : acvram/PKGBUILD et acvram/.SRCINFO"; else faux "aur : PKGBUILD ou .SRCINFO absent de l'archive"; fi
  PB=$(tar xzf "$AUR" -O acvram/PKGBUILD 2>/dev/null)
  printf '%s\n' "$PB" | grep -q "^pkgver=$V\$" && ok "aur : pkgver=$V" || faux "aur : pkgver ≠ $V"
  printf '%s\n' "$PB" | grep -q "^sha256sums=('SKIP')" && faux "aur : sha256sums=('SKIP') non renseigné" || ok "aur : sha256sums renseigné"
fi

# ---- 6. RPM : nom seulement (rpm absent sur ce poste, ou release --simule) ; sinon rpm -qip ----------------------
# Pièce 267c (CI GitHub, `rpm` présent sur le runner mais absent sur le poste de dev) : sous --simule (cette pièce
# 259, aucune release réelle) les .rpm sont de faux octets, jamais construits par rpmbuild (contrairement au .deb
# ci-dessus, bâti pour de vrai par dpkg-deb) — `rpm -qip` les rendait légitimement illisibles, uniquement là où
# `rpm` se trouve installé. Un poste sans `rpm` masquait ce défaut par accident, jamais par conception.
for r in "$RPM" "$SRPM"; do
  [ -n "$r" ] || continue
  if [ -n "$SIMULE" ]; then saute "rpm : $(basename "$r") — release simulée, nom et taille seulement"
  elif command -v rpm >/dev/null; then rpm -qip "$r" >/dev/null 2>&1 && ok "rpm : $(basename "$r") lisible" || faux "rpm : $(basename "$r") illisible"
  else saute "rpm : $(basename "$r") — rpm absent, nom et taille seulement"; fi
done

# ---- 7. traductions : archive lisible, non vide -----------------------------------------------------------------
if [ -n "$TRAD" ]; then
  N=$(unzip -l "$TRAD" 2>/dev/null | tail -n1 | awk '{print $2}')
  [ "${N:-0}" -gt 0 ] && ok "translations : $N fichiers" || faux "translations : archive vide ou illisible"
fi

# ---- 8. Flatpak : installation par le .flatpakref (dépôt OSTree + extra-data) dans une installation utilisateur SÉPARÉE,
#         puis acvram doctor dans le bac à sable — séparables (--flatpak-installer sans carte, --flatpak-doctor sous carte.sh)
if [ -n "$FLAT" ]; then
  if [ "$FLATPAK" = 0 ]; then saute "flatpak : bras non demandé (--sans-flatpak)"
  elif ! command -v flatpak >/dev/null; then saute "flatpak : commande absente"
  else
    export FLATPAK_USER_DIR="$DOSSIER/flatpak-user"      # jamais ~/.local/share/flatpak : rien sur l'installation de l'utilisateur
    mkdir -p "$FLATPAK_USER_DIR"
    grep -q '^GPGKey=' "$FLAT" && ok "flatpak : dépôt signé (GPGKey dans le .flatpakref)" || saute "flatpak : dépôt NON signé (pas de GPGKey) — --no-gpg-verify"
    if [ "$FLATPAK" != doctor ]; then
      flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo >/dev/null 2>&1 || true
      # --reinstall : un rejeu sur la même installation dédiée ne doit pas rendre FAUX pour « already installed » ;
      # 259 d : --reinstall ne traverse pas les remotes (l'app posée par l'ancien bundle vient de acvram-origin, le
      # .flatpakref propose acvram) → on retire d'abord ce qui est installé (installation dédiée : rien de l'utilisateur)
      flatpak info --user "$APP" >/dev/null 2>&1 && flatpak uninstall --user --noninteractive -y "$APP" >/dev/null 2>&1
      if flatpak install --user --noninteractive -y --reinstall --from "$FLAT" >"$DOSSIER/flatpak-install.log" 2>&1; then
        VER_FLAT=$(flatpak info --user "$APP" 2>/dev/null | sed -n 's/^ *Version: *//p' | head -n1)
        # Pièce 285 (chef, 26/09) : sur la v0.7.4, ce contrôle a dit OK en ayant installé la 0.7.2 — GitHub
        # Pages n'avait pas encore servi le nouveau dépôt OSTree au moment de l'install. Un `flatpak update`
        # attend que Pages rattrape, borné (jamais indéfiniment) ; si la version installée ne devient jamais
        # $V, c'est un REFUS net (66), jamais confondu avec un simple OK trompeur.
        if [ "$VER_FLAT" != "$V" ]; then
          echo "        flatpak : $VER_FLAT installée, $V attendue — Pages en retard, attente bornée (flatpak update)"
          i=0
          while [ "$VER_FLAT" != "$V" ] && [ "$i" -lt 10 ]; do
            i=$((i + 1))
            flatpak update --user --noninteractive -y "$APP" >>"$DOSSIER/flatpak-install.log" 2>&1 || true
            VER_FLAT=$(flatpak info --user "$APP" 2>/dev/null | sed -n 's/^ *Version: *//p' | head -n1)
            [ "$VER_FLAT" = "$V" ] || sleep "${ACVRAM_ATTENTE_FLATPAK_S:-30}"
          done
        fi
        if [ "$VER_FLAT" != "$V" ]; then
          faux "flatpak : $VER_FLAT installée après attente, $V attendue (Pages n'a jamais servi le bon dépôt)"
          echo "VERDICT: FAUX ($TAG, $DOSSIER)"
          exit 66
        fi
        ok "flatpak : installé dans $FLATPAK_USER_DIR ($VER_FLAT) — extra-data : $(flatpak run --user --command=sh "$APP" -c 'wc -l < /app/extra/apply_extra.ok' 2>/dev/null || echo '?') paquets dépaquetés"
      else faux "flatpak : installation échouée (flatpak-install.log)"; tail -n5 "$DOSSIER/flatpak-install.log" | sed 's/^/        /'; fi
    fi
    if [ "$FLATPAK" != installer ]; then
      # `flatpak run` garde le cwd de l'appelant ; lancé depuis le dépôt, acvram (_garde_arbre) refuse d'être importé depuis
      # /app quand le cwd est dans un arbre acvram — le bac à sable se lance depuis le dossier de la release
      if (cd "$DOSSIER" && flatpak run --user --command=acvram "$APP" doctor) >"$DOSSIER/flatpak-doctor.txt" 2>&1; then
        ok "flatpak : acvram doctor dans le bac à sable (flatpak-doctor.txt)"
        grep -i -E 'noyau|kernel|précompil|precompil|cuda' "$DOSSIER/flatpak-doctor.txt" | head -n6 | sed 's/^/        /'
      else faux "flatpak : acvram doctor a échoué (flatpak-doctor.txt)"; tail -n5 "$DOSSIER/flatpak-doctor.txt" | sed 's/^/        /'; fi
    fi
  fi
fi

if [ "$FAUX" = 0 ]; then echo "VERDICT: TENU ($TAG, $DOSSIER)"; exit 0; else echo "VERDICT: FAUX ($TAG, $DOSSIER)"; exit 1; fi
