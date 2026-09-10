#!/bin/bash
# Le plancher de 49,5 us : d'ou vient-il ? Protocole complet, ordre IMPOSE.
#
# L'ordre n'est pas une commodite. Chaque etape est le controle de la suivante :
#   1. empreinte  — le .so porte-t-il le sha du .cu ? ccache ment PAR LA DATE,
#                   et quatre valeurs d'un parametre ont deja rendu quatre fois
#                   le meme chiffre parce que le binaire ne changeait pas.
#   2. TEMOIN     — etape 0 (trois flottants ecrits, retour) doit couter
#                   BEAUCOUP MOINS que l'etape 3. Sinon le montage est encore
#                   faux et RIEN de ce qu'il mesure ne vaut : ARRET, sans lire.
#   3. bisection  — etapes 0..3, deux contextes.
#   4. grille     — EN DERNIER. C'est elle qui a produit la fausse refutation
#                   de la sous-parallelisation ; elle ne veut rien dire tant que
#                   le temoin n'a pas parle. L'etape 0 est REJOUEE A CHAQUE
#                   VALEUR : c'est le plancher aux bornes du balayage reel.
#
# SI LE TEMOIN NE SEPARE PAS : ne pas lire la suite, ne pas « ajuster ». C'est
# le seul cas ou le balayage K de banc_fma (toujours dans le .cu, sans cout)
# redevient necessaire — il faut alors une echelle de travail CONNUE pour
# distinguer « instrument aveugle » de « noyau vide ». En cas nominal la
# bisection EST le controle positif : elle rend deux valeurs differentes, donc
# elle prouve d'elle-meme qu'elle peut rendre autre chose.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
R="$(dirname "$S")"
PY="$R/.venv/bin/python"
M=/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram/Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4
SORTIE="${SORTIE:-/tmp/poste2-49us}"
LIMITE=1800          # aucune mesure ne tient le PC plus de 30 min
mkdir -p "$SORTIE"

dire() {   # chaque mesure previent A LA FIN DE LA MESURE, pas a la fin du lot
  echo "[$(date +%H:%M:%S)] $*"
  notify-send -a acvram "protocole 49us" "$*" 2>/dev/null || true
}
libre() {
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$u" -lt 800 ] || { dire "REFUS : carte non libre ($u Mio) — je ne prends pas la place d'une autre session"; exit 1; }
}
# med / p10 / p90 de la ligne machine du premier contexte
champ() { awk -F'\t' -v c="$2" '$1=="RESULTAT"{print $c; exit}' "$1"; }

mesure() {   # mesure <nom> <etape> <chunk> <contextes>
  libre
  local nom=$2-$3 f="$SORTIE/$1.txt"
  ACVRAM_PA_ETAPE="$2" ACVRAM_PA_CHUNK="$3" \
    timeout -k 30 $LIMITE "$PY" "$S/attn-isole.py" "$M" "$4" >"$f" 2>&1
  local c=$?
  if [ $c -ne 0 ]; then
    dire "ECHEC ($c) sur $1 — voir $f"
    tail -5 "$f"
    return 1
  fi
  grep -v '^RESULTAT' "$f" | tail -3
  dire "termine : $1 (etape $2, chunk $3)"
}

# ---- 1. empreinte ---------------------------------------------------------
libre
dire "1/4 empreinte du binaire (compilation possible, ~9 min)"
"$PY" - <<'EOF' || { echo "ARRET : le .so ne porte pas le sha du .cu"; exit 1; }
import sys
from acvram import kernels
e = kernels.get_extension()
if e is None:
    print("REFUS :", getattr(kernels, "_ERROR", "extension indisponible")); sys.exit(1)
print("empreinte : le binaire porte bien le source")
EOF
dire "termine : empreinte verifiee"

# ---- 2. temoin du montage -------------------------------------------------
dire "2/4 TEMOIN : etape 0 contre etape 3, contexte 350"
mesure temoin-e0 0 512 350 || exit 1
mesure temoin-e3 3 512 350 || exit 1
M0=$(champ "$SORTIE/temoin-e0.txt" 8); A0=$(champ "$SORTIE/temoin-e0.txt" 9); B0=$(champ "$SORTIE/temoin-e0.txt" 10)
M3=$(champ "$SORTIE/temoin-e3.txt" 8); A3=$(champ "$SORTIE/temoin-e3.txt" 9); B3=$(champ "$SORTIE/temoin-e3.txt" 10)
# Seuil ECRIT D'AVANCE, deux conditions : un ecart franc (30 %) ET un ecart plus
# grand que les dispersions cumulees. La seconde seule laisserait passer une
# separation minuscule mais reguliere ; la premiere seule laisserait passer un
# ecart franc noye dans le bruit.
VERDICT=$(awk -v m0="$M0" -v a0="$A0" -v b0="$B0" -v m3="$M3" -v a3="$A3" -v b3="$B3" \
  'BEGIN{ d=(b0-a0)+(b3-a3); print (m0 <= 0.70*m3 && (m3-m0) > d) ? "SEPARE" : "PLAT" }')
echo "temoin : etape0 = $M0 us [$A0 – $B0]   etape3 = $M3 us [$A3 – $B3]   -> $VERDICT"
if [ "$VERDICT" != "SEPARE" ]; then
  dire "ARRET : le temoin ne separe pas — le montage est encore faux, rien n'est lu"
  cat <<'EOF'
Ne pas lire la suite, ne pas ajuster le seuil apres coup. C'est ICI, et
seulement ici, que le balayage K de banc_fma devient necessaire : sans echelle
de travail connue on ne peut pas distinguer un instrument aveugle d'un noyau
vide. Restaurer le script de balayage, PUIS revenir.
EOF
  exit 2
fi
dire "termine : temoin SEPARE — le montage peut etre lu"

# ---- 3. bisection ---------------------------------------------------------
dire "3/4 bisection complete, deux contextes"
for e in 0 1 2 3; do mesure bisection-e$e $e 512 350,3000 || exit 1; done

# ---- 4. grille, EN DERNIER ------------------------------------------------
dire "4/4 grille : etape 0 rejouee a chaque valeur (plancher aux bornes reelles)"
for c in 64 128 256 512 1024 2048; do
  mesure grille-e0-c$c 0 "$c" 3000 || exit 1
  mesure grille-e3-c$c 3 "$c" 3000 || exit 1
done

echo; echo "== recapitulatif =="
awk -F'\t' '$1=="RESULTAT"{printf "%-18s ctx %5s  C %4s  %8.2f us [%.2f-%.2f]  part %s/%s\n", n, $5, $7, $8, $9, $10, $11, $12}
            FNR==1{n=FILENAME; sub(/.*\//,"",n); sub(/\.txt$/,"",n)}' "$SORTIE"/*.txt
dire "PROTOCOLE TERMINE — sorties dans $SORTIE"
