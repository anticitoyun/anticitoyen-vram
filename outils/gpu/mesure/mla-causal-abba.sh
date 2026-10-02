#!/usr/bin/env bash
# zzs : ABBA en service sur Kimi-Linear-35B, A = ACVRAM_MLA_CAUSAL=0 (témoin), B = défaut, UN arbre figé, UNE prise.
#   ACVRAM_POSTE=poste5 ACVRAM_NOM=poste5-zzs ACVRAM_TYPE=mesure ACVRAM_DUREE_MAX=2400 ATTENDU=<HEAD> \
#     outils/carte.sh bash outils/gpu/mesure/mla-causal-abba.sh
# Étape 0 : tests/test_mla_causal_zzs.py sur carte (E1(a) au bit, E2 fp32, bras cassant) ; rouge → arrêt, carte rendue.
# Puis A1 B1 B2 A2 : un serveur neuf par bras (serveur-bras.sh : PID vrai, port libre avant, VRAM rendue après), requêtes
# de mla-causal-abba.py bras ; enfin nsys p81 A puis B (nsys-kda-p81.sh inchangé, préfill 8 k au GPU ; NSYS=0 l'omet).
# Durée prédite : étape 0 ≈ 1-2 min ; par bras chargement 60-90 s + 2 chauffes + 3 × 2 requêtes à 8 k (≈ 1,4 s A) et à
# 32 k (≈ 16 s A, ≈ 9 s B) ≈ 3-3,5 min ; nsys 2 × 4-6 min ; total ≈ 25-30 min.
# Injection de test (tests/test_mla_causal_abba.py) : MLA_ABBA_LANCEUR remplace `python -m acvram serve` par un faux serveur ;
# l'étape 0 et nsys sont alors omis, et le disent.
set -euo pipefail
ICI=$(cd "$(dirname "$0")" && pwd)
ARBRE=$(cd "$ICI/../../.." && pwd)
[ "$(git -C "$ARBRE" rev-parse HEAD)" = "${ATTENDU:?ATTENDU = HEAD figé}" ] || { echo "ÉCHEC : HEAD ≠ ATTENDU"; exit 65; }
PY=${PY_ACVRAM:-$HOME/Bureau/Claude/anticitoyen-vram/.venv/bin/python}
MODELE=${ACVRAM_MODELE_MESURE:-/mnt/AI_GENERATOR/models_acvram/Kimi-Linear-35B-kda-nvfp4}
PORT=${MLA_ABBA_PORT:-8097}
NOM=kimi-zzs
SMI=${BRAS_NVIDIA_SMI:-nvidia-smi}
. "$ICI/serveur-bras.sh"
o=$(cd / && PYTHONPATH="$ARBRE" ACVRAM_ARBRE="$ARBRE" CUDA_VISIBLE_DEVICES="" "$PY" -c "import acvram, os; print(os.path.dirname(os.path.dirname(os.path.realpath(acvram.__file__))))" | tail -1)   # la garde imprime « ARBRE … » d'abord
[ "$o" = "$(realpath "$ARBRE")" ] || { echo "ÉCHEC : acvram importé depuis $o, pas de $ARBRE"; exit 3; }
O=${SORTIE:-$ARBRE/scratchpad/poste5-zzs-abba-$(date +%d-%m)}
mkdir -p "$O"
rm -f "$O"/A1.json "$O"/B1.json "$O"/B2.json "$O"/A2.json   # un bras en échec ne relit jamais une sortie antérieure
{ date +%FT%T; $SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; ps -eo pid,pcpu,comm --sort=-pcpu | head -4; } > "$O/avant.txt"
# VRAM rendue = au plus l'occupation d'avant + 1 Gio : le bras suivant ne part jamais sur une carte encore tenue
mio0=$($SMI -i "${BRAS_CARTE:-0}" --query-gpu=memory.used --format=csv,noheader,nounits)
export BRAS_VRAM_MAX_MIO=$((mio0 + 1024))
echo "== $(date +%T) arbre $ARBRE ($ATTENDU) · modèle $MODELE · VRAM avant $mio0 Mio · sortie $O"

if [ -z "${MLA_ABBA_LANCEUR:-}" ]; then
    echo "== $(date +%T) étape 0 : tests/test_mla_causal_zzs.py sur carte"
    (cd "$ARBRE" && PYTHONPATH="$ARBRE" ACVRAM_ARBRE="$ARBRE" CUDA_VISIBLE_DEVICES=0 ACVRAM_TESTS_PENDANT_MESURE=1 \
        "$PY" -m pytest -q -p no:cacheprovider -ra --tb=short tests/test_mla_causal_zzs.py) > "$O/etape0.txt" 2>&1 || true
    ! grep -qE 'SKIPPED|skipped' "$O/etape0.txt" \
        || { tail -8 "$O/etape0.txt"; echo "ÉCHEC : étape 0 sans les tests carte (sautés), ABBA non jouée"; exit 7; }
    # Scellé : E1 visé, E2 en repli. E1(a) seul rouge → l'ABBA se joue (E2 jugera) ; tout autre rouge → arrêt.
    if ! tail -1 "$O/etape0.txt" | grep -qE '^[0-9]+ passed in'; then
        if grep -E '^(FAILED|ERROR) ' "$O/etape0.txt" | grep -qv 'test_e1_au_bit_du_chemin_complet'; then
            tail -15 "$O/etape0.txt"; echo "ÉCHEC : étape 0 rouge hors E1(a), ABBA non jouée"; exit 7
        fi
        grep -q '^FAILED .*test_e1_au_bit' "$O/etape0.txt" || { tail -15 "$O/etape0.txt"; echo "ÉCHEC : étape 0 illisible"; exit 7; }
        echo "== E1(a) TOMBE sur carte (au bit) ; E2 vert : ABBA jouée, le verdict sera E2"
    fi
    tail -1 "$O/etape0.txt"
else
    echo "== étape 0 et nsys OMIS : serveur injecté ($MLA_ABBA_LANCEUR)"
fi

for bras in A1 B1 B2 A2; do
    if [ "${bras:0:1}" = A ]; then interrupteur=(env ACVRAM_MLA_CAUSAL=0); else interrupteur=(env -u ACVRAM_MLA_CAUSAL); fi
    echo "== $(date +%T) bras $bras (${interrupteur[*]})"
    if [ -n "${MLA_ABBA_LANCEUR:-}" ]; then
        # shellcheck disable=SC2086
        bras_servir "$PORT" "$O/serveur-$bras.log" "${interrupteur[@]}" $MLA_ABBA_LANCEUR "$PORT" "$NOM" || exit 70
    else
        bras_servir "$PORT" "$O/serveur-$bras.log" "${interrupteur[@]}" PYTHONPATH="$ARBRE" ACVRAM_ARBRE="$ARBRE" \
            CUDA_VISIBLE_DEVICES=0 "$PY" -m acvram serve "$MODELE" --port "$PORT" --served-name "$NOM" \
            --max-model-len 32896 --max-batch 1 --no-prefix-cache || exit 70
    fi
    bras_pret "$PORT" "$NOM" "$BRAS_PID" "${MLA_ABBA_DELAI_S:-480}" || { bras_arreter "$BRAS_PID" "$PORT" || true; exit 71; }
    (cd / && "$PY" "$ICI/mla-causal-abba.py" bras "http://127.0.0.1:$PORT" "$NOM" "$O/$bras.json" 2>&1 \
        | grep -E '^bras|Error|ÉCHEC|mla-causal-abba' | cut -c1-220) || true
    bras_arreter "$BRAS_PID" "$PORT" || { echo "ÉCHEC : arrêt du bras $bras incertain, la suite ne part pas"; exit 72; }
    [ -s "$O/$bras.json" ] || { tail -5 "$O/serveur-$bras.log"; echo "ÉCHEC : bras $bras sans sortie"; exit 1; }
done

if [ -z "${MLA_ABBA_LANCEUR:-}" ] && [ "${NSYS:-1}" != 0 ]; then
    for v in A B; do
        if [ $v = A ]; then interrupteur=(env ACVRAM_MLA_CAUSAL=0); else interrupteur=(env -u ACVRAM_MLA_CAUSAL); fi
        echo "== $(date +%T) nsys p81 $v"
        "${interrupteur[@]}" SORTIE="$O/nsys-$v" ATTENDU="$ATTENDU" bash "$ICI/nsys-kda-p81.sh" | tail -3
    done
    # preuve que l'interrupteur a pris dans le pilote nsys (sa ligne de régime), comme /metrics pour les serveurs
    grep -q 'mla_causal=0(temoin)' "$O/nsys-A/pilote.log" && ! grep -q 'mla_causal=0(temoin)' "$O/nsys-B/pilote.log" \
        || echo "AVERTISSEMENT : nsys A/B — l'interrupteur n'a pas pris comme attendu (voir pilote.log), traces non comparables"
fi
{ date +%FT%T; $SMI --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } > "$O/apres.txt"
"$PY" "$ICI/mla-causal-abba.py" comparer "$O/A1.json" "$O/B1.json" "$O/B2.json" "$O/A2.json" | tee "$O/verdict.txt"
