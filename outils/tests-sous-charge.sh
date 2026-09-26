#!/bin/bash
# Quels tests cassent VRAIMENT sous contention, et a quel taux ?
#
# Nous accusons deux tests sur la foi de deux echecs vus au hasard. Un test
# qu'on croit fragile et qui ne l'est pas, on le serialiserait pour rien.
#
# LE CONTROLE SANS LEQUEL LE RESTE NE VAUT RIEN : chaque test est joue N fois
# SANS charge d'abord. S'il echoue deja, ce n'est pas la contention qu'on
# mesure, c'est un test casse — et il est ecarte du tableau, en le disant.
set -u
S="$(cd "$(dirname "$0")" && pwd)"; R="$(dirname "$S")"
PY=${ACVRAM_PY:-$R/../../anticitoyen-vram/.venv/bin/python}
export PYTHONPATH="$R" CUDA_VISIBLE_DEVICES=0
N=${N:-5}; GIO=${GIO:-20}; CALCUL=${CALCUL:-0.5}
# DEUX EXEMPLAIRES DU MEME TRAVAIL, ET LA DUREE PUBLIEE. Une manche unique est
# INDECIDABLE : sans jumelle, rien ne peut accuser une anomalie ni la
# disculper. Mes deux premiers bras n en avaient qu un chacun — un RESISTE 5/5
# isole ne dit rien. Regle etablie par claude-f2 le 10/09 sur 47,42/47,35
# contre 60,85/67,56 : c est la comparaison des jumelles qui accuse, un fichier
# seul a 60,85 n aurait rien dit.
EXEMPLAIRES=${EXEMPLAIRES:-2}
TMP=$(mktemp)
CIBLES=${CIBLES:-"tests/test_calibration.py tests/test_fusion_nvfp4.py"}

joue() {   # joue <cible> -> "<echecs> <sauts>" sur N manches
  # UN TEST QUI SAUTE N A PAS RESISTE. pytest rend 0 quand tout est ignore :
  # compter le seul code de sortie ferait passer un saut pour un succes — et
  # sous forte charge le garde _carte_disponible fait justement sauter les
  # tests marques. On compte donc les deux, separement.
  local e=0 s=0
  for _ in $(seq 1 $N); do
    local out
    out=$(timeout -k 20 900 $PY -m pytest -q -rs "$1" 2>&1) || e=$((e+1))
    grep -qE '[0-9]+ skipped' <<<"$out" && s=$((s+1))
  done
  echo "$e $s"
}

# CAUSE ET OCCUPATION, jamais devinees. Un timeout renvoie code 124, un vrai
# echec code 1 : les traiter pareil confond « le test dure » avec « le test
# echoue », faute nommee le 11/09 sur le bras 1 (3 timeouts a 3600 s comptes
# comme echecs). L'occupation vient de pmon sm % du PID de charge, relevee
# encadrant chaque manche — sans elle, « charge peut-etre retiree ? » reste
# une question au lieu d'etre une mesure.
joue_qualifie() {   # <cible> <pid_charge> → "echecs timeouts sauts sm_min sm_moy sm_max vif_pct"
  local cible=$1 pcharge=$2 e=0 to=0 s=0
  local sm_min=100 sm_max=0 sm_somme=0 sm_n=0 vif_ok=0 vif_tot=0
  for m in $(seq 1 $N); do
    local trace=$(mktemp)
    # Echantillonneur 1 Hz : sm % du PID de charge, vivacite (kill -0). Ecrit
    # « sm|vif » par ligne. Se termine sur SIGTERM apres la manche.
    (
      while kill -0 $$ 2>/dev/null; do
        local sm=$(nvidia-smi pmon -c 1 -i 0 2>/dev/null | awk -v p="$pcharge" '$2==p{print $4}' | head -1)
        case "$sm" in ''|*[!0-9]*) sm=0 ;; esac
        local vif=0; kill -0 "$pcharge" 2>/dev/null && vif=1
        echo "$sm|$vif" >> "$trace"
        sleep 1
      done
    ) &
    local sampler=$!
    local out code
    out=$(timeout -k 20 1800 $PY -m pytest -q -rs "$cible" 2>&1)
    code=$?
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null

    # Aggregation cause + relevés
    case "$code" in
      0)   grep -qE '[0-9]+ skipped' <<<"$out" && s=$((s+1)) ;;
      124) to=$((to+1)) ;;
      *)   e=$((e+1)) ;;
    esac
    while IFS='|' read -r sm vif; do
      case "$sm" in ''|*[!0-9]*) continue ;; esac
      [ "$sm" -lt "$sm_min" ] && sm_min=$sm
      [ "$sm" -gt "$sm_max" ] && sm_max=$sm
      sm_somme=$((sm_somme + sm)); sm_n=$((sm_n + 1))
      vif_tot=$((vif_tot + 1)); [ "$vif" = 1 ] && vif_ok=$((vif_ok + 1))
    done < "$trace"
    rm -f "$trace"
  done
  local sm_moy=0
  [ "$sm_n" -gt 0 ] && sm_moy=$((sm_somme / sm_n))
  local vif_pct=0
  [ "$vif_tot" -gt 0 ] && vif_pct=$((vif_ok * 100 / vif_tot))
  echo "$e $to $s $sm_min $sm_moy $sm_max $vif_pct"
}

echo "# N=$N par cible · charge $GIO Gio, calcul $CALCUL"
declare -A sans
for c in $CIBLES; do sans[$c]="$(joue "$c")"; done

# LE VERROU, ET LE NOM QUI DIT QUE LA CHARGE EST VOULUE. Sans lui, une autre
# session voit un PID etranger qui alloue et cherche un intrus : six manches
# perdues le 10/09. `charge-gpu.py` refuse desormais de demarrer sans, donc
# cette ligne n est pas une precaution mais la seule facon de le lancer.
# --duree 86400 (24 h) : les manches lentes de test_calibration.py durent
# 3600 s sous contention ; 5 manches x 2 exemplaires x 2 cibles depassent
# largement une charge de 3600 s. Le harnais tue le groupe a la fin (trap),
# donc la duree n'est pas la borne — c'est le trap qui borne.
ACVRAM_NOM=CHARGE-DELIBEREE setsid "$S/carte.sh" \
  $PY -u "$R/outils/gpu/mesure/charge-gpu.py" --gio "$GIO" --calcul "$CALCUL" --duree 86400 \
  > "$TMP" 2>&1 &
CH=$!
# TUER LE GROUPE, PAS L ENFANT. carte.sh est l enfant, charge-gpu.py le
# PETIT-FILS : un kill sur le premier laissait le second occuper 10 Gio pendant
# la manche suivante — constate, et c est ce qui a fait echouer la charge du
# bras a 27 Gio en OOM.
trap 'kill -- -$CH 2>/dev/null; kill $CH 2>/dev/null' EXIT
sleep 10

# LA CHARGE A-T-ELLE VRAIMENT PRIS ? Sans ce controle, le tableau conclut
# « RESISTE » sur une charge qui a echoue — exactement le defaut que ce
# harnais est cense debusquer chez les autres. Constate le 10/09 : la charge
# de 27 Gio est morte en OOM et le verdict RESISTE 5/5 a ete rendu quand meme.
if ! grep -q '^charge : ' "$TMP"; then
  echo "REFUS : la charge n a pas demarre — rien a conclure sur la contention"
  sed -n '1,6p' "$TMP"
  exit 4
fi
grep '^charge : ' "$TMP"
PRIS=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
MINI=$(python3 -c "print(int($GIO*1024*0.8))")
if [ "$PRIS" -lt "$MINI" ]; then
  echo "REFUS : $PRIS Mio pris pour $GIO Gio demandes — la charge n est pas celle annoncee"
  exit 4
fi

# Trouver le PID de charge-gpu.py pour releve pmon dans la boucle des manches.
PCHARGE=$(pgrep -f 'charge-gpu\.py' | head -1)
echo "# PID charge = $PCHARGE"

printf '%-42s %8s %10s  %s\n' cible sans "avec cause" verdict
for c in $CIBLES; do
  read -r se ss <<<"${sans[$c]}"
  # EXEMPLAIRES du meme travail : deux exemplaires qui divergent accusent la
  # manche ; un exemplaire seul ne peut rien dire. Chaque exemplaire publie
  # cause (echec/timeout) et sm % relevés autour de chaque manche.
  ae=0; ato=0; as=0; sm_lo=100; sm_hi=0; vif_lo=100
  for ex in $(seq 1 $EXEMPLAIRES); do
    d0=$(date +%s)
    read -r e to s2 lo mo hi vif <<<"$(joue_qualifie "$c" "$PCHARGE")"
    printf '  exemplaire %d/%d : %s echecs, %s timeouts, %s sauts, %s s, sm %s/%s/%s %%, charge vive %s %%\n' \
      "$ex" "$EXEMPLAIRES" "$e" "$to" "$s2" "$(( $(date +%s) - d0 ))" "$lo" "$mo" "$hi" "$vif"
    ae=$((ae + e)); ato=$((ato + to)); as=$((as + s2))
    [ "$lo" -lt "$sm_lo" ] && sm_lo=$lo
    [ "$hi" -gt "$sm_hi" ] && sm_hi=$hi
    [ "$vif" -lt "$vif_lo" ] && vif_lo=$vif
  done
  # TOTAL, pas N : modifier N dans la boucle le multiplierait a chaque cible,
  # et la deuxieme cible serait comparee a un denominateur faux.
  TOTAL=$((N * EXEMPLAIRES))
  # « lent » et « échoue » sont deux choses différentes. Un timeout dit que le
  # test a été trop LONG sous contention, un échec qu'il a rendu un mauvais
  # résultat. Confondre les deux fait dire « fragile » à un test seulement plus
  # lent — faute nommée le 11/09 sur le bras 1 précédent.
  if [ "$se" -gt 0 ]; then
    v="ECARTE : echoue deja $se/$N SANS charge — test casse, pas fragile"
  elif [ "$as" -gt 0 ]; then
    v="ECARTE : a SAUTE $as/$TOTAL sous charge — un test qui saute n a pas resiste"
  # Un exemplaire dont la charge n'a pas vecu tout du long (vif_lo bas) n'a
  # pas mesure ce qu'on croit : le publier avec la meme etiquette qu'un
  # exemplaire ou vif=100 refait la faute d'etendue du 11/09.
  elif [ "$vif_lo" -lt 90 ]; then
    v="INVALIDE : charge vive $vif_lo % au pire — pas de mesure sous contention"
  elif [ "$ae" -eq 0 ] && [ "$ato" -eq 0 ]; then
    v="RESISTE $TOTAL/$TOTAL sur $EXEMPLAIRES exemplaires, sm ${sm_lo}-${sm_hi} %"
  elif [ "$ae" -eq 0 ] && [ "$ato" -gt 0 ]; then
    v="LENT sous charge : $ato timeouts / $TOTAL, 0 vrai echec, sm ${sm_lo}-${sm_hi} %"
  elif [ "$ae" -eq "$TOTAL" ]; then
    v="FRAGILE systematique $TOTAL/$TOTAL, sm ${sm_lo}-${sm_hi} %"
  else
    v="FRAGILE intermittent $ae echecs + $ato timeouts sur $TOTAL, sm ${sm_lo}-${sm_hi} %"
  fi
  printf '%-40s %5s/%d %5s/%d  %s\n' "$c" "$se" "$N" "$ae" "$TOTAL" "$v"
done
