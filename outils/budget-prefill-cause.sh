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

# UNE MANCHE QUI NE REND PAS DE LIGNE ARRETE TOUT. `carte.sh` ABANDONNE apres
# ACVRAM_ATTENTE secondes (1800 par defaut) si un autre tient la carte, et sort
# en 3 SANS rien ecrire. Sans cette garde, la boucle continuerait et la
# campagne aurait des manches manquantes — un ABBA ampute d un bras rend un
# classement, pas une absence. On compte donc la ligne, on ne la suppose pas.
manche() {  # <budget> <regime> <invite> [extra...]
  local b=$1 r=$2 inv=$3; shift 3
  local avant; avant=$(wc -l < "$SORTIE")
  CUDA_VISIBLE_DEVICES=0 ACVRAM_MAX_GRAPHS=64 \
    outils/carte.sh "$PY" outils/budget-prefill.py \
      --budget "$b" --regime "$r" --invite "$inv" "$@" \
      >> "$SORTIE" 2>>"$SORTIE.journal"
  local code=$?
  if [ "$(wc -l < "$SORTIE")" -eq "$avant" ]; then
    echo "ARRET : manche budget=$b regime=$r invite=$inv n a rendu AUCUNE ligne" >&2
    echo "        (code $code — voir $SORTIE.journal). Campagne incomplete," >&2
    echo "        rien a publier. L ABBA ampute rendrait un classement faux." >&2
    exit 4
  fi
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

# --- verdict de proprete, SEUIL POSE D AVANCE -----------------------------
# L etendue des TEMOINS d un meme point mesure la proprete de la manche, pas
# son resultat : ils font tous exactement le meme travail. Mesure sur deux
# campagnes propres : 0,06 % a 0,81 %. Le bras contamine du 10/09 : 42,7 %.
# SEUIL A 5 % — vingt fois la dispersion propre, un huitieme de la polluee.
# Un bras au-dessus n est pas « bruite », il est INVALIDE : la contention ne
# deplace pas les temoins au hasard, elle les gonfle tous du meme cote.
"$PY" - "$SORTIE" <<'FIN' >&2
import statistics as st, sys
L=[l.rstrip("\n").split("\t") for l in open(sys.argv[1])
   if l.strip() and not l.startswith("budget")]
sale=[]
for cle in sorted({(x[1], x[3]) for x in L}):
    t=[float(x[4]) for x in L if (x[1], x[3])==cle and x[0]=="0"]
    if len(t) < 2:
        continue
    e=100*(max(t)-min(t))/st.median(t)
    print(f"proprete  {cle[0]:6} invite {cle[1]:>6}  temoins {len(t)}  etendue {e:5.2f} %"
          + ("   INVALIDE" if e > 5 else ""))
    if e > 5:
        sale.append(cle)
if sale:
    print(f"ARRET DE PUBLICATION : {len(sale)} bras au-dessus du seuil de 5 % : {sale}")
    print("Les temoins font le MEME travail : une etendue pareille ne vient pas")
    print("du code mesure. Chercher un occupant de la carte avant de conclure.")
FIN
echo "[fini] manche des causes terminee, resultats dans $SORTIE" >&2
