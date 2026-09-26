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

## Résultat de l'étape 0 (prise 07:4x-07:45:14, 72ef86f5, 4 processus alternés 16/12/16/12, minimum des médianes par bras)
Contrôle N = 12 : 0,999 / 1,001 / 1,026, tenu. **T(12) / T(16) à N = 78 : 0,779 (6144 × 5120), 0,748 (17408 × 5120),
0,811 (5120 × 17408)** ; de 0,72 à 0,85 pour tout N de 16 à 78.
* **Prédiction FAUSSE dans l'autre sens.** J'attendais 1,00 à 1,10 ; la tranche 12 est 19 à 28 % plus RAPIDE. Les 2
  relectures de W en plus coûtent moins que ce que NV = 16 perd (registres ou occupation, à lire par ptxas).
* **187 telle qu'ordonnée (NV↑) : ARRÊT** (ratio < 1,15). Monter NV ralentirait encore.
* Levier opposé, au bit par construction (même ordre, cf. plus haut) : **une tranche plus petite pour 17 ≤ N ≤ 80**,
  12 ou moins, à balayer. Le témoin `ACVRAM_INT8_TRANCHE=12` existe déjà. Réserve : la tranche 16 a été posée le 14/09 pour
  le godet 16 du décodage b=12. Or ici, à N = 16, la tranche 12 (12 + 4) va aussi plus vite (0,100 contre 0,134 ms), ce qui
  est à revoir au régime du moteur (graphes, godets).

## Suite (feu de chef 07 h 5x) : balayage de la tranche par plage, écrit AVANT la prise 1
Code : `ACVRAM_INT8_TRANCHE` (N ≤ 16) et `ACVRAM_INT8_TRANCHE_PREFILL` (N > 16), valeurs 4/6/8/10/12/16, défaut 16 inchangé
(acvram_kernels.cu, `lire_tranche`). **ptxas (cuobjdump -res-usage, build de l'étape 0, bf16)** : NV 4 = 99 registres,
5-6 = 128, 8-9 = 168, 10 = 205, 12 = 217, 16 = 254, 0 déversement. À 256 fils, cela fait 2 blocs par SM pour NV ≤ 6 et 1 bloc
de NV 7 à 16. J'ajoute donc 4 et 6 au balayage (8, 10, 12 ordonnés).
* **Prise 1, banc isolé** (3 formes, N 12/16/32/64/78, 2 passes par tranche) : meilleure tranche contre 16 à N = 78,
  **−20 à −35 %** ; godet 16, **−20 à −30 %**. Prédit : les tranches 4-6 (2 blocs par SM) devant 8-12, sans certitude
  (4 lectures de W de plus à N = 78). **FAUX** si aucune tranche ne passe sous −15 % à N = 78.
* **Test au bit** (`tests/test_int8_tranche_187.py`) : couche 0 réelle du mixte (qkv, gate, out), N 2-80, sorties bf16 et
  fp32. Prédit : 6/6 au bit contre la tranche 16. Cassant (prise 2) : ordre des mots inversé pour NV ≤ 12 → ROUGE.
  **FAUX** si un seul écart hors cassant.
* **Prise 3, servi** : la meilleure tranche de chaque plage, choisie par le banc isolé selon cette règle écrite ici
  (plus petit temps à N = 78 pour le préfill, à N = 16 pour le décodage), ABBA A = 16/16, B = choisie.
  * banc chat b=8, mixte : **+2 à +5 %**. Les int8 du préfill passent au GEMV environ 3 456 fois par passage, part du
    préfill non mesurée. **FAUX** sous +1 %.
  * banc chat b=8, Qwen3.8-27B-nvfp4 : **0 ± bruit** (aucun int8 servi au préfill, 183). Témoin nul ; > 2 σ = instrument.
  * décodage b=16, mixte : **0 à +3 %**, selon la part de `int8_gemv` au godet 16 (les i8c à M ≤ 16 prennent aussi les
    chemins étroits, kernels/__init__.py:1366-1392). Compteur `CHEMINS_INT8` relevé dans la prise.

## Résultat de la prise 1 (07:57:11-08:00:27, 9534dd6d)
Balayage (minimum de 2 passes, écart à la tranche 16) : **la tranche 6 est la meilleure partout** : N = 78 −21,9 / −37,8 / −43,4 % ;
N = 16 −23,2 / −36,8 / −43,0 % ; N = 12 −3,4 / −21,9 / −23,4 % (6 + 6 fait mieux que 12 d'un coup). Tranche 4 : −23 à −39 %,
tranche 8 : −1 à −28 %, tranches 10 et 12 : −15 à −36 %. La prédiction (−20 à −35 % à N = 78) tient, et le haut de la fourchette
est dépassé sur 17408 × 5120 (−43 %). Les 2 blocs par SM (NV ≤ 6) l'emportent sur les relectures de W.
Test au bit : **6/6 verts** (couche 0 réelle du mixte, N 2-80, bf16 et fp32). Un échec : `test_toute_variable_de_chemin…`,
parce que `lire_tranche` recevait le nom de la variable et que le scanner ne voit que `getenv("ACVRAM_…")` littéral.
Corrigé, à rejouer en prise 2.
**Choix par la règle écrite avant : B = 6/6** (décodage et préfill).
