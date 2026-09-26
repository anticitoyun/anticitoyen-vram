# Verdict — bras Marlin dense (101) sur Qwen3.8-27B à M = 8, micro-banc par forme projeté au pas — 24/09 02 h 3x (poste1)

* **instrument** : `scratchpad/poste1-p128-24-09/banc-marlin-dense-q38.py` (poids NVFP4 aléatoires au format de l'alias, L copies distinctes par forme, un graphe CUDA, médiane de 20 rejeux ; chemin compté par `CHEMINS_NVFP4`) ; prise `prise-banc.sh`
* **commit** : 9124e127 (poste1-mtp)
* **régime** : RTX 5090, -lgc 2700 posé et rendu ; M = 8 ; formes et nombres par pas lus au manifeste de `Qwen3.8-27B-nvfp4` (gate‖up empilé 34 816 × 5 120 comme dans la 126) ; cpu-safe=off (max_perf_pct 100/100) ; compute-apps début = fin (llama-server 4627, sur la 3080 Ti)
* **scellé** : `scratchpad/poste1-p128-24-09/scelle-marlin-dense.md` (commits 6b9451de puis 02c7cdf3, avant la mesure) — alarme : projection `etroit` à ± 20 % de la 126 ; gain ≥ 4,7 ms (≥ 50 % de l'écart à NInfer) → bascule 101 pour les denses ; < 2 ms → 128 W4A4
* **mesuré** (µs par appel, To/s effectifs ; `etroit` = gemm_dense_etroit, `marlin` = port Marlin dense) :

| forme | N × K | par pas | etroit µs (To/s) | marlin µs (To/s) |
|---|---|---|---|---|
| gate‖up | 34 816 × 5 120 | 65 | 117,86 (0,85) | **65,16 (1,54)** |
| down | 5 120 × 17 408 | 65 | 64,57 (0,78) | **33,67 (1,49)** |
| GDN qkv | 10 240 × 5 120 | 48 | 35,12 (0,84) | 22,12 (1,33) |
| GDN gate | 6 144 × 5 120 | 48 | 25,57 (0,69) | 15,82 (1,12) |
| GDN out / o_proj | 5 120 × 6 144 | 48 + 17 | 24,1 (0,73) | 14,52 (1,22) |
| q_proj | 12 288 × 5 120 | 17 | 47,69 (0,74) | 25,21 (1,40) |
| k/v_proj | 1 024 × 5 120 | 17 + 17 | 7,35 (0,40) | **10,1 (0,29)** — plus lent |
| tête | 248 320 × 5 120 | 1 | 920,8 (0,78) | 436,6 (1,64) |

  Projection par pas : **etroit 18,32 ms, marlin 10,40 ms, gain 7,92 ms**. Alarme : 18,32 contre 21,1 mesurés (−13 %),
  gate‖up 7,66 contre 9,08 ms (−16 %) → **dans les ± 20 %**, le banc représente le pas. Justesse : erreur relative max
  par ligne identique dans les deux bras (0,0034-0,0043, arrondi bf16 de la sortie) — même arithmétique W4A16.
* **verdict** : **gain ≥ 4,7 ms → la bascule 101 revient pour les modèles denses**, dans les termes du scellé. Ramené à
  l'échelle de la 126 (× 21,1 / 18,32), le gain vaut ≈ **9,1 ms** : le pas passerait de 26,2 à ≈ 17,1 ms, contre
  16,8 pour NInfer. **Le Marlin W4A16 reprend à lui seul l'essentiel de l'écart à NInfer, sans quantifier les
  activations.** Ma prédiction resserrée (gain de 4,7 à 7,0 ms) était trop PRUDENTE : Marlin tient 1,5 To/s sur les
  formes du MLP, plus que les 1,1-1,5 prévus. Hors du chemin : k/v_proj (N = 1 024), plus lents en Marlin → seuil
  N ≥ 2 048.
* **durée** : 02:34:35 → 02:34:52 (17 s de banc)

## Ce que ce verdict ne dit pas, et la suite (au chef)
1. **b=1** : le Marlin ne sert pas à M = 1 (`_PROJ_MARLIN_MIN_M` = 2). Tel quel, la 101 garde les poids naturels
   POUR le GEMV, en DOUBLE → + 14,6 Go → le modèle entier ne tient pas (OOM prévu). Il faut une **disposition
   unique** : Marlin seul, avec un GEMV à M = 1 qui lit la disposition Marlin (`nvfp4_gemv_marlin`, déjà servi aux
   experts en E = 1). Mesure préalable, courte : GEMV naturel contre GEMV Marlin à M = 1 sur ces formes. À la 100a,
   notre GEMV gagnait 21 % à b=1 sur Coder : si la perte à b=1 dépasse ~5 %, il faut arbitrer.
2. **Modèle entier** : ms/pas ABBA ≥ 5 lots à b=8 et b=1, KL b=1 contre le chemin actuel, après la disposition unique.
3. **128 (W4A4)** : son gain à venir ne porte plus que sur l'écart restant (≈ 0,3 ms si la projection tient), au prix
   de la qualité. **Je propose de la geler.**
