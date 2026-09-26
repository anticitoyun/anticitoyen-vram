#!/usr/bin/env bash
# Serie deterministe : cinq passages, empreinte du texte, regime releve A CHAQUE
# passage. Ecrit d'apres docs/SERIE-DETERMINISME.md.
#
# POURQUOI LE REGIME A CHAQUE PASSAGE, ET PAS UNE FOIS AU DEBUT
# Le 8 septembre 2026, le bridage des cartes a change EN COURS de journee
# (775 W avant 18h20, 875 W apres 18h26) et personne ne s'en est apercu sur le
# moment : c'est la colonne plafond_W du banc, relevee a chaque ligne, qui a
# permis de dater le changement APRES COUP. Un regime lu une seule fois au
# demarrage aurait laisse croire a une serie homogene.
#
# POURQUOI L'ETAT DE LA MACHINE A CHAQUE PASSAGE
# Le meme jour, un processus tiers a avale 14 Go de memoire vive a 25 Mo par
# seconde et tue plusieurs mesures ; les echecs ont d'abord ete attribues a la
# contention entre sessions, a tort. Une serie qui ne sait pas dire ce que la
# machine faisait pendant qu'elle mesurait ne peut pas se defendre.
set -u

MODELE="${1:?usage : serie-determinisme.sh <modele> [passages] [invite]}"
PASSAGES="${2:-5}"     # cinq et non trois : avec trois, un premier passage froid
                       # et deux passages proches donnent la meme image qu'une
                       # bimodalite, et on ne les distingue pas
INVITE="${3:-Explique en trois phrases ce qu est la memoire virtuelle.}"
SORTIE="${SORTIE:-serie-determinisme-$(date +%Y%m%d-%H%M%S).tsv}"
SEUIL_RSS_GO="${SEUIL_RSS_GO:-20}"

# RSS du plus gros processus, en gigaoctets, et son nom.
plus_gros_rss() { ps -eo rss --sort=-rss | sed -n 2p | awk '{printf "%.1f", $1/1048576}'; }
plus_gros_nom() { ps -eo comm --sort=-rss | sed -n 2p; }

# Somme des plafonds de puissance des cartes, et etat de persistance.
plafond_w()   { nvidia-smi --query-gpu=power.limit --format=csv,noheader | awk '{s+=$1} END {printf "%.0f", s}'; }
persistance() { nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | sed -n 1p | tr -d ' '; }

printf 'passage\tjetons\tduree_s\tdebit_t_s\tempreinte\tplafond_W\tpersistance\trss_max_Go\tproc_max\tmem_dispo_Go\n' > "$SORTIE"
echo "### modele   : $MODELE"
echo "### sortie   : $SORTIE"
echo "### passages : $PASSAGES"

for i in $(seq 1 "$PASSAGES"); do
    rss_go=$(plus_gros_rss)
    proc_max=$(plus_gros_nom)
    depasse=$(awk -v r="$rss_go" -v s="$SEUIL_RSS_GO" 'BEGIN {print (r>s) ? 1 : 0}')
    if [ "$depasse" = "1" ]; then
        echo "### ARRET au passage $i : $proc_max occupe $rss_go Go, seuil $SEUIL_RSS_GO."
        echo "### La machine n est pas calme ; une serie mesuree ainsi ne se defend pas."
        exit 2
    fi

    plaf=$(plafond_w)
    pers=$(persistance)
    dispo=$(free -g | awk 'NR==2 {print $7}')

    debut=$(date +%s.%N)
    texte=$(acvram_generer "$MODELE" "$INVITE" 2>/dev/null)
    fin=$(date +%s.%N)

    duree=$(awk -v a="$debut" -v b="$fin" 'BEGIN {printf "%.3f", b-a}')
    jetons=$(printf '%s' "$texte" | wc -w)
    debit=$(awk -v j="$jetons" -v d="$duree" 'BEGIN {if (d>0) printf "%.2f", j/d; else printf "0"}')
    emp=$(printf '%s' "$texte" | sha256sum | cut -c1-16)

    printf '%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$i" "$jetons" "$duree" "$debit" "$emp" "$plaf" "$pers" "$rss_go" "$proc_max" "$dispo" >> "$SORTIE"
    echo "passage $i : $jetons jetons, $debit j/s, empreinte $emp, $plaf W, $pers"
done

# La dispersion se lit sur la mediane et l'etendue, jamais sur le meilleur
# passage : le meilleur-de-N est un estimateur biaise qui favorise
# mecaniquement le moteur le plus disperse.
awk -F'\t' '
NR>1 { n++; d[n]=$4+0; e[$5]=1; if (p!="" && p!=$6) melange=1; p=$6 }
END {
    if (n==0) { print "### aucun passage"; exit }
    for (i=1;i<n;i++) for (j=i+1;j<=n;j++) if (d[j]<d[i]) { t=d[i]; d[i]=d[j]; d[j]=t }
    med = (n%2) ? d[int((n+1)/2)] : (d[n/2]+d[n/2+1])/2
    printf "\n### debit median %.2f j/s, etendue %.2f a %.2f, dispersion %.1f %%\n", \
           med, d[1], d[n], (med>0) ? 100*(d[n]-d[1])/med : 0
    k=0; for (x in e) k++
    printf "### empreintes distinctes : %d sur %d passages\n", k, n
    if (k > 1) print "### LES TEXTES DIFFERENT : defaut de determinisme a temperature nulle."
    else       print "### textes identiques : la dispersion ne vient pas du texte."
    if (melange) print "### ATTENTION : le plafond de puissance a CHANGE pendant la serie."
}' "$SORTIE"
