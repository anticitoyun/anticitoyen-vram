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
