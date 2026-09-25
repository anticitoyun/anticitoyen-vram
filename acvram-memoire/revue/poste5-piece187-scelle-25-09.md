# Scellé — pièce 187 : GEMV int8 à NV 32 (ou plus) pour 17 ≤ M ≤ 80 (poste5, 25/09, AVANT toute mesure)

Ordre de chef : ptxas → test au bit → banc isolé → ABBA sur le mixte et Qwen3.8. Prédiction de départ, reprise de mon
message : préfill de 0,7 à 0,8 s par lot, banc de +5 à +8 %.

## Prémisse corrigée avant de coder (lecture du code)
* La tranche vaut **déjà 16** (acvram_kernels.cu:1727-1731 ; `ACVRAM_INT8_TRANCHE=12` rend l'ancien découpage). À M = 78, W est
  donc lu **5 fois**, pas 10 comme je l'avais écrit à chef. Le docstring de `int8_matmul` (« tranche de 8 ») est périmé.
* Poids : 6144 × 5120 font 31 Mo, soit ≈ 0,02 ms par lecture à 1,8 To/s. Or le GEMV mesure 0,41 ms à 64 jetons (docstring).
  Les relectures de W ne pèsent donc que quelques pour cent de ce temps.
* Activations : chaque bloc de ROWS = 4 lignes relit les N lignes de x. Sur 6144 × 5120 à N = 78, le trafic L1/L2 de x
  vaut ≈ 1 536 × 78 × 5 120 × 2 ≈ 1,2 Go, et il dépend de ROWS, **pas de NV**. Les registres (acc et part [ROWS][NV],
  x4 [NV][4]) font 192 à NV = 16 ; NV = 32 déborderait à ROWS = 4.
* Au bit : chaque sortie (r, n) accumule dans `part[r][n]` avec j croissant, puis `block_reduce_rows` réduit ligne par
  ligne (cu:151-165). L'ordre ne dépend ni de NV ni de ROWS. Le test au bit restera obligatoire.

## Étape 0 (≤ 3 min de carte, sans code) : `banc-tranche.py`, tranche 12 contre 16
Trois formes (6144 × 5120, 17408 × 5120, 5120 × 17408), N ∈ {12, 16, 32, 64, 78}, médiane de 50 lancements, un processus par bras.
* Si les relectures de W payaient, T(12) / T(16) à N = 78 vaudrait ≈ 7/5 = 1,40. **Prédit : 1,00 à 1,10.**
* **Arrêt de la 187** (NV sans levier ; prédiction ≤ 5 % du GEMV) si le ratio est < 1,15 sur les trois formes. Je proposerai
  alors ROWS↑, qui réduit le trafic de x, en pièce à part. **On continue** (ptxas, puis NV = 32 à ROWS réduit) si le ratio
  est > 1,25. Entre les deux, je le dis sans conclure et la décision revient à chef.
* Contrôle de prise : à N = 12, les deux bras font un seul lancement, donc T(12) / T(16) doit valoir 1,00 ± 0,05. Sinon
  la tranche n'a pas pris, ou l'instrument ment.
