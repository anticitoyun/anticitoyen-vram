# Protocole — voyants de `narrow_gemm` (Sage § 6) : b=1 non régressé, ties à l'ulp, chiffre b=12

Laure, 15/09/2026, avant mesure. Ordre : Sage § 6
[`sage-reprise-15-09-b.md`](sage-reprise-15-09-b.md) (main 77b5def), transmis par
Jérôme. Moteur : main du jour (`ACVRAM_NARROW_GEMM` défaut 0, garde
`kernels/__init__.py:501`, `:681` pour l'int8 ; NVFP4 étroit coupé `:505`).
Manon fait la PPL teacher-forcing (1,000 ± 0,002) ; moi les trois voyants
ci-dessous, une prise de carte (~15 min).

## (1) b=1 non régressé — ABAB NARROW=0/1, témoin int8, 20 s

Seuil Sage : **B ≤ 4,30 ms** (témoin 4,297) ; réfuté > 4,38 → narrow
conditionné à M ≥ 4. Ma prédiction : `_NARROW_MIN` = 2 (`:502`) → à b=1 le
chemin étroit n'est même pas pris (n=1 < 2) : **B = A à ±0,01 ms** ;
réfuté si B ≠ A de plus de 0,02 (alors la garde ne fait pas ce qu'elle dit).

## (2) Ties b=12 — critère d'Océane à l'ulp

Même montage que c6377d5 (12 × 256 jetons, b=12 tenu, graphes), bras A =
NARROW=0, B = NARROW=1, MMA au défaut (godet 12). Capture : top-2 + amax
par position, **lignes complètes de logits (f16) pour les 64 premiers
jetons** de chaque séquence. Verdict par `acvram.quant.equivalence.verdict_position`
(delta = max|A−B| sur la ligne, échelle = max|A|, cos, top-1, écart
top1-top2 de A ; multiplicateur du module = **5 ulp**, recalé par les
témoins d'Océane — Jérôme écrit « 2 ulp », le module dit 5 : j'applique le
module et je publie les écarts en ulp pour que Sage tranche). Conforme si
**aucune position hors ex-aequo n'échoue** ; une seule = bogue, pas ON.
Ma prédiction : les 4 divergences de Laurine (s4@31, s7@2, s8@1, s11@5) sont
des ex-aequo à ≤ 2 ulp (même arithmétique, ordre des sommes seul) ; 0
position en échec sur 12 × 64. Réfuté si une position est hors ex-aequo
(alors narrow change plus que l'ordre des sommes : à Laurine).

## (3) Chiffre officiel b=12 après l'ON — ABAB NARROW=0/1, protocole c6377d5

Rondes ctx 2048, invite 256, 20-31 s, repos 30 s, une carte, température.
Attendu (Sage) ≈ **9,5 ms / 0,31 J** (Laurine 22 s : 9,52 / 0,310 / 391 W).
Ma prédiction : A = 15,0-15,6 ms (v0.6.5 route+pack, jamais tamponné par
moi) ; **B = 9,6-10,2 ms** (mes rondes incluent le prefill et la traîne de
lot, +2-5 % sur le banc court), J 0,30-0,33. Réfuté si B > 10,5 (le banc
court mentait) ou < 9,3.
