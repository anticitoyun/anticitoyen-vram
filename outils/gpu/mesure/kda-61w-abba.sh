#!/usr/bin/env bash
# 61w : ABBA sur Kimi-Linear-35B, deux arbres figés (A = main avant la fusion de poste5-kda, B = la fusion), UNE prise ≤ 30 min.
#   ACVRAM_POSTE=poste1 ACVRAM_NOM=poste1-61w ACVRAM_TYPE=mesure ACVRAM_DUREE_MAX=1500 \
#     ARBRE_A=… ATTENDU_A=… ARBRE_B=… ATTENDU_B=… outils/carte.sh bash outils/gpu/mesure/kda-61w-abba.sh
# Chaque bras : un processus neuf, PYTHONPATH sur SON arbre, acvram.__file__ contrôlé ; arrêt au premier bras en échec.
# Durée prédite : 4 × (chargement 60-90 s + 2 × 2 048 pas ≈ 30 s + chrono ≈ 5 s) ≈ 8-9 min.
set -euo pipefail
ICI=$(cd "$(dirname "$0")" && pwd)
PY=${PY_ACVRAM:-$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python}
export ACVRAM_KERNELS_PRECOMPILES=${ACVRAM_KERNELS_PRECOMPILES:?dossier des .so précompilés de A ET de B (aucun JIT pendant la prise)}
export ACVRAM_MODELE_MESURE=${ACVRAM_MODELE_MESURE:-/mnt/AI_GENERATOR/models_acvram/Kimi-Linear-35B-kda-nvfp4} CUDA_VISIBLE_DEVICES=0
for v in A B; do
    a=$(eval echo "\${ARBRE_$v:?}"); h=$(eval echo "\${ATTENDU_$v:?}")
    [ "$(git -C "$a" rev-parse HEAD)" = "$h" ] || { echo "ÉCHEC : arbre $v $(git -C "$a" rev-parse --short HEAD) ≠ $h"; exit 65; }
    o=$(cd / && PYTHONPATH="$a" ACVRAM_ARBRE="$a" CUDA_VISIBLE_DEVICES="" "$PY" -c "import acvram, os; print(os.path.dirname(os.path.dirname(os.path.realpath(acvram.__file__))))" | tail -1)   # la garde imprime « ARBRE … » d'abord
    [ "$o" = "$(realpath "$a")" ] || { echo "ÉCHEC : acvram de $v importé depuis $o"; exit 3; }
done
O=${SORTIE:-$ARBRE_B/scratchpad/poste1-61w-$(date +%d-%m)}
mkdir -p "$O"
rm -f "$O"/A1.json "$O"/B1.json "$O"/B2.json "$O"/A2.json   # un bras en échec ne relit jamais une sortie antérieure
{ date +%FT%T; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; ps -eo pid,pcpu,comm --sort=-pcpu | head -4; } > "$O/avant.txt"
# Étape 0 (chef) : le test au bit de 61w sur carte, dans l'arbre B (D=128 jamais joué sur carte). Rouge → arrêt, carte rendue.
echo "== $(date +%T) étape 0 : tests/test_kda_etat_kv.py (arbre B)"
(cd "$ARBRE_B" && PYTHONPATH="$ARBRE_B" ACVRAM_ARBRE="$ARBRE_B" ACVRAM_TESTS_PENDANT_MESURE=1 "$PY" -m pytest -q -p no:cacheprovider \
    -ra --tb=short tests/test_kda_etat_kv.py) > "$O/etape0.txt" 2>&1 || { tail -15 "$O/etape0.txt"; echo "ÉCHEC : étape 0 rouge, ABBA non jouée"; exit 7; }
tail -1 "$O/etape0.txt"
for bras in A1 B1 B2 A2; do
    v=${bras:0:1}; a=$(eval echo "\$ARBRE_$v")
    echo "== $(date +%T) bras $bras ($(git -C "$a" rev-parse --short HEAD))"
    (cd / && PYTHONPATH="$a" ACVRAM_ARBRE="$a" "$PY" "$ICI/kda-61w-abba.py" bras "$O/$bras.json" 2>&1 \
        | grep -E 'régime|^bras|PRECOMPILE|Error|ÉCHEC|kda-61w' | cut -c1-220) || true
    [ -s "$O/$bras.json" ] || { echo "ÉCHEC : bras $bras sans sortie"; exit 1; }
done
{ date +%FT%T; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$O/apres.txt"
"$PY" "$ICI/kda-61w-abba.py" comparer "$O/A1.json" "$O/B1.json" "$O/B2.json" "$O/A2.json" | tee "$O/verdict.txt"
