#!/bin/bash
# Pièce 48 — où part le débit à M = 12 : trois bras ncu sur les noyaux EN L'ÉTAT
# (aucun code modifié). Prédictions et seuils : revue/poste1-piece48-ncu-etroites-22-09.md
#
#   outils/carte.sh outils/gpu/mesure/ncu-etroites-p48.sh <commit-attendu> [sortie]
#
# Compteurs réservés à root (ERR_NVGPUCTRPERM) : relancer la MÊME ligne avec
# NCU_SUDO=1 — le sudo est appliqué à `ncu` seul, à l'intérieur. Ne JAMAIS
# écrire `sudo outils/carte.sh …` : cela casserait le verrou pour tout le groupe.
#
# Bras A `_dense_etroit_kernel` (nvfp4) et C `nvfp4_gemv_marlin_kernel` sur
# l'alias alpha2 ; bras B `etroit`/`gemm_etroit` int8 sur l'alias officiel.
# ncu est lent (replay par noyau) : `--launch-count` borne chaque bras, les
# métriques sont ciblées — `--set full` coûterait la fenêtre entière.
set -uo pipefail
cd "$(dirname "$0")/../../.."
ATTENDU=${1:?usage: ncu-etroites-p48.sh <commit-attendu> [sortie]}
D=${2:-scratchpad/poste1-p48-ncu}
[ "$(git rev-parse --short HEAD)" = "$ATTENDU" ] || {
  echo "REFUS : HEAD $(git rev-parse --short HEAD) != $ATTENDU" >&2; exit 65; }
mkdir -p "$D"
export PYTHONPATH=$PWD
PY=$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python
# `ncu` n'est pas dans le PATH des sessions (CUDA hors PATH) : le premier jeu
# de bras a rendu rc=127 quatre fois sans rien mesurer. Chemin explicite, et
# refus AVANT la prise plutôt qu'un bras vide qu'on prendrait pour un résultat.
NCU=${NCU:-/usr/local/cuda/bin/ncu}
command -v "$NCU" >/dev/null || { echo "REFUS : ncu introuvable ($NCU)" >&2; exit 66; }

# Le sudo est ICI, autour de `ncu` seulement — JAMAIS autour de `carte.sh`.
# Sous `sudo outils/carte.sh …`, les fichiers du verrou (/tmp/acvram-carte-0.lock,
# .qui, .journal) seraient recréés root : plus aucune session ne pourrait écrire
# son `.qui`, et le verrou serait cassé pour tout le groupe, pas seulement pour
# la prise en cours. Même raison pour les sorties : les redirections `>` sont
# évaluées par CE shell, non par root, donc les CSV restent à l'utilisateur.
# `HOME` est forcé parce que sudo le réécrit en /root selon `always_set_home` :
# sans lui, le python du venv recompilerait l'extension dans /root/.cache
# (2 min 30 sous la fenêtre, et un cache que l'utilisateur ne peut plus purger).
SUDO=""
if [ "${NCU_SUDO:-0}" = "1" ]; then
  sudo -n true 2>/dev/null || {
    echo "REFUS : NCU_SUDO=1 mais sudo demande un mot de passe — un script qui" >&2
    echo "  attend une saisie sous le verrou bloque la carte pour tout le monde." >&2
    echo "  Faire valider le sudo hors fenêtre, puis relancer." >&2; exit 78; }
  SUDO="sudo -n -E HOME=$HOME PYTHONPATH=$PYTHONPATH"
fi
PARC=${ACVRAM_MODELES:-$(python3 outils/racine_modeles.py)}
ALPHA2=$PARC/Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha2-22-09
OFFICIEL=$PARC/Qwen3-Coder-30B-A3B-nvfp4
M="smsp__issue_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
sm__inst_executed.sum,\
gpu__time_duration.sum"

# Les compteurs de performance NVIDIA sont réservés à root par défaut
# (NVreg_RestrictProfilingToAdminUsers=1) : sans ce contrôle, les quatre bras
# chargent le modèle, tournent cinq minutes et rendent ERR_NVGPUCTRPERM — cinq
# minutes de carte pour zéro métrique (22/09, première prise de la 48). Le
# micro-lancement ci-dessous coûte trois secondes et refuse AVANT la fenêtre.
echo "== contrôle des compteurs (ERR_NVGPUCTRPERM)"
if ! $SUDO "$NCU" --metrics dram__bytes.sum --csv \
        $PY -c "import torch; torch.zeros(1024, device='cuda').sum().item()" 2>&1 \
        | tee "$D/controle-compteurs.log" | grep -q "dram__bytes"; then
  echo "REFUS : compteurs ncu inaccessibles — voir $D/controle-compteurs.log" >&2
  grep -o "ERR_NVGPUCTRPERM" "$D/controle-compteurs.log" | head -1 >&2
  echo "  remède (engage la machine, décision de l'utilisateur) :" >&2
  echo "  NVreg_RestrictProfilingToAdminUsers=0 dans /etc/modprobe.d/99-optim-nvidia.conf," >&2
  echo "  puis update-initramfs -u et redémarrage ; ou lancer ce script sous sudo." >&2
  exit 77
fi

echo "== compute-apps début $(date +%H:%M:%S)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
echo "== charge hôte"; ps -eo pid,pcpu,comm --sort=-pcpu | head -4

bras () {                       # $1 nom, $2 alias, $3 regex de noyau, $4 lancements
  echo "== bras $1 ($3) $(date +%H:%M:%S)"
  ACVRAM_MODELE_MESURE="$2" $SUDO "$NCU" --graph-profiling node -k regex:"$3" \
      --launch-count "$4" --metrics "$M" --csv \
      $PY outils/gpu/mesure/frontiere-pas.py "$D/$1.json" 12 8 \
      > "$D/$1.csv" 2> "$D/$1.log"
  local rc=$?
  local n=$(wc -l < "$D/$1.csv" 2>/dev/null || echo 0)
  echo "   rc=$rc  lignes=$n"
  # un bras qui ne mesure rien n'est pas un bras : le dire ici, pas au résumé
  [ "$rc" -eq 0 ] && [ "$n" -lt 2 ] && { echo "   BRAS VIDE — 3 dernières lignes du log :"; tail -3 "$D/$1.log"; }
  # ncu refuse parfois le profilage sous graphe : le dire, ne pas le masquer
  grep -qi "graph" "$D/$1.log" && grep -i "graph" "$D/$1.log" | head -2
  return 0
}

bras A_nvfp4_etroit   "$ALPHA2"  "_dense_etroit_kernel"     6
bras B_int8_etroit    "$OFFICIEL" "etroit|gemm_etroit"      6
bras C_marlin_avec    "$ALPHA2"  "nvfp4_gemv_marlin_kernel" 4
bras C_marlin_sans    "$OFFICIEL" "nvfp4_gemv_marlin_kernel" 4

echo "== compute-apps fin $(date +%H:%M:%S)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
echo "== résumé"
$PY outils/gpu/mesure/ncu-resume-p48.py "$D"
