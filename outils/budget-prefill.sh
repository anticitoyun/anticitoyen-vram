#!/bin/bash
# Protocole de la manche du budget de prefill — FIXE AVANT D'AVOIR VU UN CHIFFRE.
#
#     outils/budget-prefill.sh <sortie.tsv> [budget...]
#
# TROIS DECISIONS PRISES D'AVANCE, ET LA RAISON DE CHACUNE
# --------------------------------------------------------
# 1. ACVRAM_MAX_GRAPHS=64 DANS TOUS LES BRAS, y compris le bras temoin
#    budget=0. A 16 — le defaut — la campagne de Laure a mesure 65,11 % de
#    dispersion ; a 64, 20,37 %. Ce n'est pas un reglage qu'on compare ici :
#    c'est du BRUIT qu'on retire des deux cotes a la fois. Le poser d'un seul
#    cote fabriquerait l'ecart qu'on cherche.
#
# 2. SEUIL DE CONCLUSION : 20 %. Un ecart de budget inferieur a la dispersion
#    residuelle n'est PAS departageable, et sera rapporte « non etabli », pas
#    « faible ». Ecrit ici pour ne pas se negocier plus tard avec ce qu'on
#    esperait trouver.
#
# 3. ABBA, pas A puis B. La derive thermique vaut +3 W en douze passages et
#    biaise l'ordre ; alterner simplement ne la corrige pas, ABBA si.
#
# LE RESULTAT APPARTIENT AU REGIME 64. Il ne se transporte pas au produit,
# dont le defaut est 16 — a 16 places un prefill decoupe rencontre plus de
# formes et pese sur un cache deja sature, sans eviction : le decoupage peut
# etre gagnant a 64 et perdant a 16. SI LE DEFAUT RESTE A 16, CETTE MANCHE
# EST A REFAIRE A 16 AVANT DE POSER UN BUDGET PAR DEFAUT.
#
# LE REGIME QUI DECIDE EST `charge`. `seule` ne peut que montrer une perte —
# c'est le prix du decoupage. Si le prix depasse ce que `charge` rachete, le
# defaut RESTE 0.
#
# Une valeur par processus : le balayage dans un seul processus fragmenterait
# la carte pour les essais suivants.
set -u
cd "$(dirname "$0")/.."
SORTIE=${1:?usage: budget-prefill.sh <sortie.tsv> [budget...]}
shift
BUDGETS=("$@"); [ ${#BUDGETS[@]} -gt 0 ] || BUDGETS=(0 2048 4096 8192)
PY=.venv/bin/python

printf 'budget\tregime\tinvite\tlatence_prefill_ms\tpasses_prefill\t' > "$SORTIE"
printf 'jetons_voisines\tjetons_s_voisines\tjoules\tjetons_par_kJ\twatts\t' >> "$SORTIE"
printf 'temp_max\tbridages\tinvalidations\n' >> "$SORTIE"

manche() {  # <budget> <regime>
  CUDA_VISIBLE_DEVICES=0 ACVRAM_MAX_GRAPHS=64 \
    outils/carte.sh "$PY" outils/budget-prefill.py \
      --budget "$1" --regime "$2" >> "$SORTIE" 2>>"$SORTIE.journal"
}

for regime in seule charge; do
  for b in "${BUDGETS[@]}"; do
    [ "$b" = 0 ] && continue
    # A B B A : le temoin encadre le bras, la derive porte sur les deux.
    manche 0 "$regime"; manche "$b" "$regime"
    manche "$b" "$regime"; manche 0 "$regime"
  done
done
echo "[fini] manche du budget terminee, resultats dans $SORTIE" >&2
