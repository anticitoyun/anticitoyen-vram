#!/usr/bin/env bash
# 11e (poste6, 30/09) — PPL appariée des échelles de bloc nvfp4 : max6 (AbsMax, défaut) / 4sur6 / balayage (ScaleSweep)
# sur Qwen2.5-Coder-7B-Instruct, trois conversions IDENTIQUES sauf --echelle, puis un `acvram eval` par converti sur le
# MÊME corpus et les MÊMES fenêtres, et le bootstrap apparié (ppl-appariee-bootstrap.py). Scellé (prédiction, seuils,
# issues) : acvram-memoire/revue/poste6-11e-scelle-ppl-30-09.md — ÉCRIT AVANT, à relire avant de lancer.
#
# Quatre prises ≤ 30 min, chacune sous outils/carte.sh (verrou = seule vérité sur la carte, ACVRAM_DUREE_MAX 1800),
# ACVRAM_POSTE=poste6 ; nvidia-smi --query-compute-apps relevé au début ET à la fin de chaque prise ; HEAD asserté.
#   ppl-balayage-11e.sh convertir max6      (prise 1)     ppl-balayage-11e.sh convertir 4sur6     (prise 2)
#   ppl-balayage-11e.sh convertir balayage  (prise 3)     ppl-balayage-11e.sh evaluer             (prise 4 : 3 evals)
#   ppl-balayage-11e.sh analyser                          (à sec : bootstrap apparié, aucune carte)
# Sorties sous $SORTIE (défaut : scratchpad/poste6-11e-ppl-<date>/), JSON d'eval + journaux + relevés.
set -uo pipefail
ICI=$(dirname "$(readlink -f "$0")"); DEPOT=$(cd "$ICI/../../.." && pwd)
# Addendum poste1 01/10 : commit figé du jour obligatoire, mesureur ≠ auteur (poste6 a écrit ScaleSweep), carte 0 seule,
# témoin max6 rejoué au bit, MAX_TOKENS 65536 → 262144 si l'IC95 dépasse ± 0,25 % (NON RÉSOLU).
COMMIT_ATTENDU="${COMMIT_ATTENDU:?COMMIT_ATTENDU = HEAD figé du jour (addendum du scellé)}"
MAX_TOKENS="${MAX_TOKENS:-65536}"
SOURCE="${SOURCE:-/mnt/4TO_SATACMR_2022/Modeles/models/Qwen2.5-Coder-7B-Instruct}"
CORPUS="${CORPUS:-/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt}"
CORPUS_SHA="e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
PROFIL="${PROFIL:-rig-14900k-5090-3080ti}"
RACINE="${ACVRAM_MODELES:-/mnt/AI_GENERATOR/models_acvram}"
SORTIE="${SORTIE:-$DEPOT/scratchpad/poste6-11e-ppl-$(date +%Y%m%d)}"
PY="${PY:-$DEPOT/.venv/bin/python}"
export ACVRAM_POSTE="${ACVRAM_POSTE:?poste du mesureur}" ACVRAM_DUREE_MAX="${ACVRAM_DUREE_MAX:-1800}" CUDA_VISIBLE_DEVICES=0
[ "$ACVRAM_POSTE" != poste6 ] || { echo "ÉCHEC : l'autrice de ScaleSweep ne mesure pas sa méthode (REGLES § 3)"; exit 5; }
mkdir -p "$SORTIE"

nom_de() { echo "Qwen2.5-Coder-7B-11e-$1"; }        # trois dossiers frères, même source, même profil, même calibration
releve() { { date +%FT%T; uptime; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader;
             nvidia-smi --query-gpu=power.limit,clocks.sm --format=csv,noheader; } >> "$SORTIE/releve-$1.txt" 2>&1; }
verifier_tete() {
  local head; head=$(git -C "$DEPOT" rev-parse --short=9 HEAD)
  if [[ "$head" != "$COMMIT_ATTENDU"* && "$COMMIT_ATTENDU" != "$head"* ]]; then
    echo "ÉCHEC : HEAD $head ≠ $COMMIT_ATTENDU (scellé)"; exit 2; fi
  echo "# HEAD $head · poste $ACVRAM_POSTE · durée max ${ACVRAM_DUREE_MAX}s · sortie $SORTIE"
}

case "${1:-}" in
  convertir)
    ech="${2:?convertir <max6|4sur6|balayage>}"; verifier_tete
    dest="$RACINE/$(nom_de "$ech")"
    [ -e "$dest" ] && { echo "ÉCHEC : $dest existe déjà (pas d'écrasement)"; exit 3; }
    releve "debut-convertir-$ech"
    "$DEPOT/outils/carte.sh" "$PY" -m acvram.cli convert "$SOURCE" --profile "$PROFIL" -o "$dest" --echelle "$ech" \
      2>&1 | tee "$SORTIE/convertir-$ech.log"; rc=${PIPESTATUS[0]}
    releve "fin-convertir-$ech"
    echo "# convertir $ech : rc $rc ($(grep -c . "$SORTIE/convertir-$ech.log") lignes)"; exit "$rc" ;;
  evaluer)
    verifier_tete
    [ "$(sha256sum "$CORPUS" | cut -c1-64)" = "$CORPUS_SHA" ] || { echo "ÉCHEC : corpus ≠ sha256 du scellé"; exit 4; }
    releve "debut-evaluer"; rc=0
    for ech in max6 4sur6 balayage max6-t; do          # max6-t : témoin de l'instrument, max6 rechargé et rejoué
      d="$RACINE/$(nom_de "${ech%-t}")"; [ -f "$d/acvram_manifest.json" ] || { echo "ÉCHEC : $d absent"; exit 3; }
      "$DEPOT/outils/carte.sh" "$PY" -m acvram.cli eval "$d" --corpus "$CORPUS" --window 2048 --stride 2048 \
        --min-context 0 --max-tokens "$MAX_TOKENS" --device cuda:0 --json > "$SORTIE/eval-$ech.json" 2> "$SORTIE/eval-$ech.err" \
        || { rc=$?; echo "eval $ech : rc $rc — série arrêtée"; releve "fin-evaluer"; exit "$rc"; }
    done
    releve "fin-evaluer"; exit "$rc" ;;
  analyser)
    { "$PY" - "$SORTIE/eval-max6.json" "$SORTIE/eval-max6-t.json" <<'PYEOF'
import json, sys
a, b = (json.load(open(c)) for c in sys.argv[1:3])
a, b = (x[0] if isinstance(x, list) else x for x in (a, b))
ok = a["par_fenetre"] == b["par_fenetre"] and len(a["par_fenetre"]) > 0
print("TÉMOIN max6 = max6-t AU BIT" if ok else "TÉMOIN max6 ≠ max6-t : S2-S4 INVALIDES (bruit d'instrument)")
PYEOF
      "$PY" "$ICI/ppl-appariee-bootstrap.py" "$SORTIE/eval-max6.json" "$SORTIE/eval-4sur6.json" "$SORTIE/eval-balayage.json" --tirages 20000
      "$PY" "$ICI/ppl-appariee-bootstrap.py" "$SORTIE/eval-4sur6.json" "$SORTIE/eval-balayage.json" --tirages 20000
      # contrôle gratuit : la conversion max6 neuve contre l'alias Qwen2.5-Coder-7B-nvfp4 existant (mêmes options ?)
      for f in "$RACINE/Qwen2.5-Coder-7B-nvfp4"/acvram-0000*.safetensors; do sha256sum "$f" | cut -c1-16; done | sort | md5sum
      for f in "$RACINE/$(nom_de max6)"/acvram-0000*.safetensors; do sha256sum "$f" | cut -c1-16; done | sort | md5sum
    } | tee "$SORTIE/analyse.txt" ;;
  *) sed -n '2,12p' "$0"; exit 64 ;;
esac
