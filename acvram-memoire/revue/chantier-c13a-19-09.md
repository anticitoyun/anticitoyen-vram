# Chantier C13-a — TF32 à portée limitée sur le cœur d'attention MLA de GLM (à sec, 25 min) ; à mesurer par poste2

Objectif (poste7, `poste7-c7-clos-c13-attention-glm-19-09`) : le cœur d'attention MLA tourne en fp32 plein (`mla.py` : einsum scores q·C, o_lat = probs·V), sans TF32 — 8,6 TFLOP par pas de prefill GLM, ≥ 82 ms des 372 ms ; les tensor cores TF32 (entrées 10 bits de mantisse, > bf16 ; accumulation fp32) sont 8× plus rapides que le fp32 CUDA-core.

Prédiction scellée (poste7, recopiée) : pas de prefill GLM 372 → **315-325 ms** ; scellé (poste2) : PPL = défaut **± 0,001** ; prefill **≥ 6 200 j/s** (5 502 au défaut).

## Le geste

`acvram/engine/mla.py` : `ACVRAM_MLA_TF32=0|1` (défaut 0, `regime.VARIABLES`, liste de garde), contexte `_tf32_coeur()` qui pose `torch.backends.cuda.matmul.allow_tf32 = True` **pendant les deux einsum du cœur seulement** (scores et o_lat, chemin chunké par 256 requêtes et chemin non chunké : 4 sites) et restaure le drapeau après — la portée est le cœur, pas le processus ; `v_b`, `k_b`, les projections et tout le reste gardent leur précision. Rien ne change au défaut.

## Preuve à sec (`tests/test_mla_tf32_c13.py`, 2 tests)

TF32 émulé (mantisse tronquée à 10 bits = borne haute de l'arrondi matériel) sur les formes de GLM (20 têtes, rank 512 + rope 64, 2 047 clés) : écart au fp32 plein **< 2⁻¹⁰ relatif** sur scores, o_lat et y (v_b réel de la couche 3 du converti calibA lu depuis le disque) ; témoin cassant : bf16 (7 bits) dépasse la borne. Contexte : drapeau posé puis restauré ; identité au défaut ; les 4 sites comptés dans le source.

## Reste / non vérifié

* Le temps réel (allow_tf32 sur `einsum` → cuBLAS TF32 sur sm_120 : à vérifier par nsys que les noyaux `*tf32*` apparaissent, sinon la variable est inerte — REGLES § 4, contrôle qui peut rendre faux).
* La PPL sous TF32 (poste2 : `ppl-acvram` GLM 3 tranches, `[gMASK]<sop>`, contre le défaut du même arbre).
* C13-b (cœur bf16 sur noyau) après le nsys GLM.

## C13-b (19/09 soir, à sec, commit `poste1-11 6bd11a13`) — `ACVRAM_MLA_CORE=fp32|tf32|bf16` remplace le booléen `ACVRAM_MLA_TF32`

Geste (`poste7-m2-mma2-budgets-prefill-19-09` § 3) : une seule variable de régime, trois bras. `mla.py:48-70` : `_MLA_CORE` lu une fois (valeur hors {fp32, tf32, bf16} → `ValueError` à l'import), `_dt_coeur()` (bf16 sous `bf16`, fp32 sinon), `_tf32_coeur(decode=)` inchangé mais armé seulement sous `tf32`. Les opérandes des trois produits du cœur (scores q·C, o_lat = probs·V, y = v_b·o_lat) sont transtypés en `_dt_coeur()` aux 5 sites de préfill (`mla.py:351-375`) et aux 4 sites de décodage (`mla.py:341, 485`) ; la sortie est refloatée avant la suite — la capture graphes ne change pas (mêmes tenseurs d'entrée/sortie, un cast de plus dans le graphe). `regime.py` : `Variable("MLA_CORE", "fp32", …)` ; `cli.py` : garde mise à jour. Hors défaut, `regime_ligne()` imprime `ACVRAM_MLA_CORE=bf16`.

Preuve à sec (`tests/test_mla_tf32_c13.py`, 3 tests, 32 passed / 23 skipped sur les 8 fichiers MLA/régime) : `test_coeur_bf16_borne_a_2_moins_7_par_ligne_formes_reelles` — formes GLM réelles (nh 20, r 512, rope 64, **L = 2047**, t = 64 requêtes), v_b réel du converti si présent sinon aléatoire fixé une fois pour les deux bras, bras bf16 contre fp32 : |Δ| ≤ 2⁻⁷ · max|ligne| sur **chaque ligne** des scores, de o_lat et de y ; **bras qui doit différer** : une ligne de y perturbée de 2⁻⁶ fait rendre « faux » à la borne. `test_contexte_tf32_restaure_et_identite_au_defaut` : sous `fp32` le contexte ne touche pas `allow_tf32`, sous `tf32` il le pose puis le restaure.

Reste (carte, poste2) : bras `bf16` scellé avant mesure par poste7 — prefill GLM **≥ 9 000 j/s ET ΔPPL ≤ +0,002** → devient le défaut ; sinon `tf32` (C13-a) reste le candidat. Non vérifié à sec : la vitesse (bf16 double le débit des tensor cores contre TF32, prédiction ≥ +5 % sur le préfill GLM si les 82 ms du cœur sont bien la part dominante) ; l'effet sur la PPL en décodage (2 sites, teacher forcing b=1).
