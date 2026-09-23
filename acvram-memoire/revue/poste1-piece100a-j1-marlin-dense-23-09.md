# Pièce 100 A, J1 — Marlin DENSE nvfp4 (vLLM 0.29) contre notre int8 servi : ARRÊT (2 cellules sur 4) — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-proj-nvfp4.py`. L2 froid (48 poids distincts), graphe de 48 appels, mur/48, médiane de 30 ; CUPTI en information ; justesse de chaque bras contre fp64.
  * bras vLLM : `apply_fp4_marlin_linear`, groupe 16, réduction fp32, atomiques coupées ;
  * bras acvram : `backends.matmul` sur de l'int8 symétrique PAR CANAL, comme l'alias qkvo-i8c.
* **commit** : 1289f677 ; vLLM 0.29.0 (Triton 3.7.1) contre acvram (Triton 3.8.0), confondu nommé.
* **régime** : -lgc 2700, horloge moyenne sous charge 2 691 MHz ; au début et à la fin, seul llama-server 4627 tourne ; prise de 7 s (17:37:19-26).
* **scellé** : `scratchpad/poste1-p100a-23-09/scelle-j1.md`, commité avant la prise (9e0c5d16).
* **mesuré** (µs par couche ; chemin int8 compté) :

| cellule | int8 servi (chemin) | Marlin nvfp4 | rapport | seuil ≤ 0,85 | prédit int8 / Marlin |
|---|---|---|---|---|---|
| qkv [5 120, 2 048] M=12 | 10,10 (`etroit_triton`) | **7,59** | 0,751 (−25 %) | tenu | 9,2-10,5 / 4,5-6,5 |
| qkv M=1 | 8,76 (`gemv`) | **7,28** | 0,831 (−17 %) | tenu | 7-11 / 4,0-6,5 |
| o [2 048, 4 096] M=12 | 11,29 (`etroit_triton`) | **9,62** | **0,852** (−14,8 %) | **raté de 0,002** | 9,3-12 / 4,0-6,0 |
| o M=1 | 7,60 (`gemv`) | **8,60** | **1,13 (+13 %)** | **raté** | 6-10 / 3,5-6,0 |

  * Poids : int8 10,5 / 8,4 Mo contre nvfp4 5,9 / 4,7 Mo. Débit effectif du Marlin : 0,78 To/s (qkv) et 0,49 To/s (o).
  * Justesse : Marlin 2,4-2,7e-3, int8 1,6-1,7e-3 contre fp64 ; les deux bras calculent ce qu'ils disent.
* **verdict** : **ARRÊT**, au scellé : le GO demandait les quatre cellules.
  1. Mes prédictions pour le Marlin étaient trop optimistes, de 30 à 140 %. J'avais appliqué au noyau la moitié des octets ; or il reste loin de la bande sur ces formes étroites, à 0,5-0,8 To/s. La pire est `o`, avec N = 2 048 et K = 4 096.
  2. À M=1, notre `int8_gemv` (lecture pure) bat le Marlin sur `o`. C'est l'issue nommée au scellé, et elle frappe le cas b=1, celui de l'écart d'énergie à llama.cpp.
  3. Mes octets comptés, le Marlin lit 44 % de moins que l'int8 sans être 44 % plus rapide.
* **Non mesuré, hors scellé** : l'énergie.
  * À temps presque égal, lire deux fois moins d'octets peut quand même baisser les J/jeton. La prémisse de la 99, « l'écart d'énergie vient des octets », n'est pas jugée ici : ce banc ne juge que le temps.
* **Voie non jouée, à la connaissance du chef** : notre noyau `gemm_dense_etroit.py` (W4A16 nvfp4 à petit M, Triton, déjà servi pour l'alias à projections nvfp4 de la p42) n'a pas été mis au même banc. Ce serait un bras de 5 s, à sceller avant.
* **durée** : ≈ 45 min ; carte 7 s.
