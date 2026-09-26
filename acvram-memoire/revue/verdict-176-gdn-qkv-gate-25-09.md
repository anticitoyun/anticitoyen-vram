# Verdict — 176 : GDN qkv‖gate INT8 en une pile au décodage — 25/09 (poste1)

* **code** (branche poste1-176) : `stack_int8_linears([qkv, gate])` dans `GatedDeltaNet.fuse` (vues, aucune copie), servie à
  M ≤ 16 ; `gemm_etroit._etroit_segments_kernel` : tables gpt et tranches PAR TUILE, chaque segment garde la partition K
  que `decouper_k` lui donne seul (qkv [14,14,12], gate [10×4]) ; grille (256, 4), programme au-delà des tranches de sa
  tuile sorti aussitôt ; préfill en deux appels sur les vues ; témoin ACVRAM_GDN_QKV_GATE=0 (VARIABLES_LUES, regime).
* **commit** : 32fd773e ; prises poste1-p176-tests, -mixte-b8, -mixte-b1, -qwen38-b8, -qwen38-b1 ; scellé
  `scratchpad/poste1-p176-25-09/scelle.md` (avant).
* **au bit** : test_gdn_qkv_gate_176 (5 M × {par canal, g128}, projections GDN à M = 1, 8, 64) + test_gemm_etroit,
  test_regime_noyaux, test_cadrage_perplexite : 54 verts, 1 sauté ; témoin « partition commune » différent (montage) ;
  cassants ROUGES : `_segments` ignoré dans gemm_etroit → 8 rouges ; pile sans `_segments` → 2 rouges.
* **ABBA** (certifie-b12, CERT_PUR, CTX 8 192, CIBLE 15 s, -lgc 2700, 5 lots par bras, écart entre lots ≤ 0,3 %) :

| alias | b | A (2 appels) | B (pile) | B/A | prédit | verdict |
|---|---|---|---|---|---|---|
| mixte (i8c) | 8 | 19,735 ms | 19,670 ms | **0,9967** | 0,975-0,995 | **FAUX sur l'ampleur** (−0,065 ms au lieu de −0,25), sans régression |
| mixte (i8c) | 1 | 15,343 ms | 15,133 ms | **0,9863** | 0,985-1,000 | TENU (−0,21 ms) |
| Qwen3.8 nvfp4 (témoin) | 8 | 13,775 | 13,774 | 0,9999 | 0,995-1,005 | TENU |
| Qwen3.8 nvfp4 (témoin) | 1 | 12,447 | 12,446 | 0,9999 | 0,995-1,005 | TENU |

* **lecture** : à b=1 le gain (1,4 %) dépasse la prédiction : un GEMV et sa montée de moins par couche. À b=8 le modèle
  « coût fixe + octets / 1,56 To/s » surestimait l'économie : la pile ne gagne que ≈ 1,4 µs par couche sur 65,7. Causes
  possibles, non mesurées : les 160 programmes vides et les lectures de tables, une grille de 1 024 programmes (1,7 vague)
  dont la seconde vague est à moitié pleine, un pipeline de lecture moins bon à N = 16 384. À trancher par un nsys si la
  suite (grille de gate‖up, PDL) le demande.
* **sortie servie** : inchangée au bit.
