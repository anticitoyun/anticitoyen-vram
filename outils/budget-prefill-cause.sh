#!/bin/bash
# D OU VIENT LE +50,7 % D ENERGIE DU PREFILL DECOUPE — et ce que le lot change.
#
#     outils/budget-prefill-cause.sh <sortie.tsv>
#
# DEUX MECANISMES PREDISENT LE MEME SIGNE, DONC AUCUN CHIFFRE GLOBAL NE LES
# SEPARE :
#
#   relecture des poids   cout FIXE par passe : quatre passes relisent
#                         quatre fois ce qu une passe lit une fois
#   efficacite des GEMM   un GEMM de 8192 lignes est plus efficace qu un de
#                         2048, et la courbe s APLATIT — donc un surcout qui
#                         DECROIT a chaque division supplementaire
#
# CE QUI LES SEPARE : la difference A INVITE EGALE.
#
#     D(L, tranche) = E(L decoupe) - E(L en une passe)
#
# A invite egale le travail d attention est IDENTIQUE des deux cotes — meme
# masque causal, meme total. Verifie par le calcul, pas suppose :
#
#     8192 en une passe   8192^2/2                       = 33,55 M
#     8192 en 4 tranches  6 x 2048^2 + 2 x 2048^2        = 33,55 M
#
# Le contexte qui grandit compense exactement ce que chaque tranche a de plus
# court. D annule donc l attention et ne laisse que les deux mecanismes.
#
# UN CONTROLE ECARTE : « quatre prefills INDEPENDANTS de 2048 ». Il ne fait
# que 8,4 M unites d attention contre 33,55 — 25 % du travail. Il serait
# tombe sous la mesure decoupee QUOI QU IL ARRIVE : un controle qui ne peut
# pas rendre « faux ».
#
# LES PREDICTIONS, POSEES AVANT LA MESURE :
#
#     D(8192, tranche 2048) = 132,5 J          deja mesure, 3 passes en plus
#     cout fixe par passe   -> D(4096) ~ 44 J  (le tiers)
#     efficacite des GEMM   -> D(4096) ~ 70-76 J, fortement sur-lineaire
#
#     SEUIL : D(4096) > 60 J refute le cout fixe par passe.
#
# Le point a tranche 1024 (8 passes) donne la FORME de la courbe ; il ne
# tranche pas la cause, `passes` et `taille` y variant encore ensemble.
#
# REGIME `lot` : douze invites prefillees ensemble. Si un cout fixe par passe
# existe, il est partage par les douze et le surcout par jeton est divise
# d autant. Ce bras peut PRECISER le verdict « le defaut reste 0 » — il vaut
# pour une invite longue SEULE — sans le contredire.
set -u
cd "$(dirname "$0")/.."
SORTIE=${1:?usage: budget-prefill-cause.sh <sortie.tsv>}
PY=.venv/bin/python

printf 'budget\tregime\tplaces_graphes\tjetons_prefilles\tlatence_prefill_ms\t' > "$SORTIE"
printf 'passes_prefill\tjetons_voisines\tjetons_s_voisines\tjoules\tjetons_par_kJ\t' >> "$SORTIE"
printf 'watts\ttemp_max\tbridages\tinvalidations\n' >> "$SORTIE"

manche() {  # <budget> <regime> <invite> [extra...]
  local b=$1 r=$2 inv=$3; shift 3
  CUDA_VISIBLE_DEVICES=0 ACVRAM_MAX_GRAPHS=64 \
    outils/carte.sh "$PY" outils/budget-prefill.py \
      --budget "$b" --regime "$r" --invite "$inv" "$@" \
      >> "$SORTIE" 2>>"$SORTIE.journal"
}

# --- bras A : le point qui tranche la cause, tranche 2048, ABBA ------------
for _ in 1 2; do
  manche 0 seule 4096 ; manche 2048 seule 4096
  manche 2048 seule 4096 ; manche 0 seule 4096
done

# --- bras B : la forme de la courbe, 8 passes -----------------------------
manche 0 seule 8192 ; manche 1024 seule 8192
manche 1024 seule 8192 ; manche 0 seule 8192

# --- bras C : le lot, qui peut preciser le verdict ------------------------
# Quatre invites de 8192 et non douze : douze feraient 98 304 jetons de
# cache KV, soit ~14 Gio sur ce modele, et un forward unique de cette taille
# au bras temoin. Quatre suffisent a poser la question — un cout fixe par
# passe partage par quatre doit voir son surcout par jeton divise par quatre.
# L invite DOIT depasser le budget, sinon la tranche vaut l invite entiere et
# le bras est nul sans le dire.
for b in 0 2048; do
  manche "$b" lot 8192 --lot 4 --max-model-len 16384
  manche "$b" lot 8192 --lot 4 --max-model-len 16384
done

echo "[fini] manche des causes terminee, resultats dans $SORTIE" >&2
