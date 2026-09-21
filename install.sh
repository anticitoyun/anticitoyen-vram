#!/usr/bin/env bash
# Installateur acvram. Choisit la version de torch adaptee aux GPU presents.
set -euo pipefail

cd "$(dirname "$0")"
VENV="${ACVRAM_VENV:-.venv}"
PY="${PYTHON:-python3}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die() { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# ---- python -----------------------------------------------------------------
command -v "$PY" >/dev/null || die "python3 introuvable"
"$PY" - <<'EOF' || die "acvram exige Python 3.10 ou plus recent"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
EOF
say "python: $("$PY" --version)"

# ---- choix de l'index torch -----------------------------------------------------
# Le sm_120 (Blackwell) ne tourne pas sur une roue compilee pour CUDA < 12.8.
INDEX=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    say "capacites de calcul presentes : ${CAPS//$'\n'/, }"
    if grep -qE '^(12|10)\.' <<<"$CAPS"; then
        INDEX="https://download.pytorch.org/whl/cu130"
        NVCC_WHEEL=1
        say "Blackwell detecte -> installation de torch pour CUDA 13.0"
    else
        INDEX="https://download.pytorch.org/whl/cu124"
    fi
else
    warn "pas de nvidia-smi ; installation de torch pour processeur"
    INDEX="https://download.pytorch.org/whl/cpu"
fi

# ---- environnement virtuel -------------------------------------------------------------------
if [ ! -d "$VENV" ]; then
    say "creation de $VENV"
    "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip wheel

say "installation de torch depuis $INDEX"
pip install --quiet --index-url "$INDEX" torch

# cccl : sans lui le nvcc des roues pip n a pas nv/target (cuda_fp16.h:4492
# fatal error, 19/09, doctor « repli noyaux de reference » sur le .deb 0.6.13) ;
# nvidia-cuda-cccl>=13,<14 verifie par Sage dans le venv du paquet. EPINGLE 13.0.* :
# sans borne, pip prend nvcc 13.4 contre le runtime 13.0 de torch cu130 et cccl
# refuse (« CUDA compiler and CUDA toolkit headers are incompatible », doctor
# 19/09 18 h 05 sur la 3080 Ti) ; 13.0.88 + 13.0.85 compilent sm_86 et sm_120.
# Le nvcc de la distribution est souvent trop ancien pour emettre du sm_120
# (Mint 22.3 livre CUDA 12.0). On prend celui des roues pip, que
# acvram/kernels/__init__.py sait trouver tout seul.
if [ "${NVCC_WHEEL:-0}" = 1 ]; then
    say "installation de nvcc (roues cuda-toolkit)"
    pip install --quiet --only-binary=:all: 'cuda-toolkit[nvcc,cccl]==13.0.*' \
        || warn "nvcc non installe ; les noyaux fusionnes retomberont sur la reference"
fi

say "installation d'acvram"
# extra vision inclus par defaut : transformers + pillow, indispensables a la tour
# multimodale (sans eux, un modele vision echoue au chargement et P3 casse).
pip install --quiet -e '.[dev,vision]'

# ---- verification -----------------------------------------------------------------
say ""
acvram doctor || warn "doctor a signale des problemes ; voir ci-dessus"
acvram convert --help >/dev/null 2>&1 || warn "acvram convert indisponible ; la gui (acvram-gui) ne pourra pas convertir"
say ""
say "activez avec :  source $VENV/bin/activate"
say "puis essayez :   acvram detect"

# ---- outils terminal (Claude/Kimi/tout agent) : optionnel -------------------
# acvram installe le paquet Python ; ce bloc, active seulement sur demande,
# depose EN PLUS les lanceurs qui routent Claude Code / Kimi Code (ou tout
# autre agent en terminal) vers acvram et les moteurs voisins (vLLM,
# llama.cpp, TabbyAPI) — kimi-modeles/claude-modeles (choix de modele),
# acvram-serveur (demarrage du serveur choisi), etc. Source : config-ia/bin,
# le depot qui suit ces scripts (voisin d'anticitoyen-vram sur cette machine ;
# ACVRAM_OUTILS_SRC pour en pointer un autre). Ne fait rien par defaut : ne
# doit jamais bloquer une installation qui ne veut que le paquet acvram.
if [ "${1:-}" = "--outils-terminal" ] || [ "${ACVRAM_INSTALL_OUTILS:-0}" = 1 ]; then
    SRC="${ACVRAM_OUTILS_SRC:-$(dirname "$(pwd)")/config-ia/bin}"
    DEST="$HOME/.local/bin"
    say ""
    say "outils terminal : deploiement depuis $SRC vers $DEST"
    if [ ! -d "$SRC" ]; then
        warn "source introuvable ($SRC) ; rien deploye. Fixez ACVRAM_OUTILS_SRC."
    else
        mkdir -p "$DEST"
        # Ecartes : chemins fixes sur un ancien $HOME (migration du 13/09,
        # pas remappes) — les deployer casserait plutot que reparer.
        ECARTES="memoire memoire-consolider memoire-sync exporter-projet-ia installer-projet-ia"
        # Par defaut, un fichier deja present dans DEST n'est PAS ecrase : un
        # deploiement en masse a deja effacé une fois une correction locale
        # (18/09, kimi-modeles) avec une copie plus ancienne de SRC. Utiliser
        # ACVRAM_OUTILS_FORCE=1 pour ecraser volontairement (ex. apres avoir
        # syncronise SRC en premier).
        n_copies=0; n_ecartes=0; n_gardes=0
        for f in "$SRC"/*; do
            [ -f "$f" ] || continue
            nom="$(basename "$f")"
            case " $ECARTES " in
                *" $nom "*) n_ecartes=$((n_ecartes + 1)); continue;;
            esac
            if [ -e "$DEST/$nom" ] && [ "${ACVRAM_OUTILS_FORCE:-0}" != 1 ]; then
                n_gardes=$((n_gardes + 1)); continue
            fi
            cp -a "$f" "$DEST/$nom"
            chmod +x "$DEST/$nom"
            n_copies=$((n_copies + 1))
        done
        say "  $n_copies script(s) depose(s) dans $DEST"
        [ "$n_gardes" -gt 0 ] && say "  $n_gardes deja present(s), gardes tels quels (ACVRAM_OUTILS_FORCE=1 pour ecraser)"
        [ "$n_ecartes" -gt 0 ] && warn "  $n_ecartes ecarte(s) (chemin fige sur l'ancien home, a corriger a la main : $ECARTES)"
    fi

    # ---- catalogue de modeles (acvram-chemins.tsv) : controle, pas fabrication --
    # kimi-modeles/claude-modeles et acvram-serveur resolvent un alias via ce
    # TSV (dossier, contexte). Une migration de disque ou de home peut le
    # laisser pointer sur un montage qui n'existe plus (trouve le 18/09 :
    # 195/201 lignes sur un prefixe perime apres la migration du 13/09,
    # 98 autres deplacees sur un second disque de modeles). Ce bloc CONTROLE
    # et signale ; il ne devine jamais un nouveau chemin a la place de
    # l'utilisateur — un alias sans dossier reel est un modele a reconvertir
    # ou une ligne a retirer, pas quelque chose a corriger seul.
    TSV="${ACVRAM_CHEMINS_TSV:-$HOME/TSV/acvram-chemins.tsv}"
    if [ -f "$TSV" ]; then
        say ""
        say "catalogue de modeles : controle de $TSV"
        "$PY" - "$TSV" <<'PYEOF'
import sys
tsv = sys.argv[1]
total = manquants = 0
orphelins = []
with open(tsv, encoding="utf-8") as f:
    for ligne in f:
        ligne = ligne.rstrip("\n")
        if not ligne.strip():
            continue
        total += 1
        champs = ligne.split("\t")
        if len(champs) < 2:
            continue
        alias, dossier = champs[0], champs[1]
        import os
        if not os.path.isdir(dossier):
            manquants += 1
            orphelins.append(alias)
if manquants:
    print(f"  {manquants}/{total} alias sans dossier reel (chemin casse ou modele absent)")
    for a in orphelins[:10]:
        print(f"    - {a}")
    if manquants > 10:
        print(f"    ... et {manquants - 10} de plus")
else:
    print(f"  {total}/{total} alias resolvent un dossier reel")
PYEOF
    fi
fi

# Éco d'horloge par défaut (sage-eco-2700-defaut-19-09 § 4) : le service pose
# `sudo -n nvidia-smi -lgc 2700,2700` au chargement et le rend à l'arrêt. Le droit
# sudo est un acte de l'utilisateur sur l'arbre de dev : la ligne est imprimée,
# jamais installée ici (le .deb la pose dans /etc/sudoers.d/acvram-nvidia-smi).
if command -v nvidia-smi >/dev/null 2>&1 && ! sudo -n -l nvidia-smi >/dev/null 2>&1; then
    echo
    echo "  eco 2700 : pas de droit sudo -n sur nvidia-smi — le service tournera a l'horloge libre"
    echo "  (eco=2700(libre: refus sudo), aucune cellule publiable). Pour l'accorder :"
    echo "    sudo visudo -f /etc/sudoers.d/acvram-nvidia-smi"
    echo "    $(id -un) ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -i 0 -lgc 2700\\,2700, /usr/bin/nvidia-smi -i 0 -lgc 2100\\,2100, /usr/bin/nvidia-smi -i 0 -rgc"
    echo "  (formes exactes : ce sudo refuse les jokers dans les arguments ; ajouter -i 1 pour la seconde carte)"
    echo "  puis : acvram doctor (verifie le droit ET l'effet)."
fi
