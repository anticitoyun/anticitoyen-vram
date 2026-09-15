# Verdict — pré-échelle 2^k pour down_proj, GLM (16/09)

poste1, à sec (aucun GPU), ordre poste7 §4 (`revue/poste7-glm-pile-correctif-16-09.md`,
main c50f12f), suite au plancher dénormal confirmé sur `down_proj`
(`revue/verdict-glm-saturation-16-09.md`, 23-31 % de blocs à zéro).

Script `outils/controle-plancher-echelle-16-09.py`, résultat complet
`/tmp/glm-discriminateur-mma0/resultat-plancher-echelle.json`.

## Protocole

Même `act / s_down` (8 experts, 12288 blocs poolés) que le contrôle
précédent. `k_act` = plus grand `k` tel que `max(amax) × 2^k ≤ 672`
(marge sous la borne dure du noyau, 6×448÷4). Fraction de blocs dont
`amax < 0,0117 / 2^k` pour k = 6, 8, 10, 12 et k_act.

## Mesuré

`max(amax) = 0,28857291`, `p99,9(amax) = 0,15551493`.

**`k_act = 11`** (`max×2^11 = 590,997 ≤ 672` ; `max×2^12 = 1181,995 > 672`).

| k | seuil (0,0117/2^k) | fraction restante |
|---|---|---|
| 6 | 1,828×10⁻⁴ | 0,000000 % |
| 8 | 4,570×10⁻⁵ | 0,000000 % |
| 10 | 1,143×10⁻⁵ | 0,000000 % |
| **11 (k_act)** | 5,713×10⁻⁶ | **0,000000 %** |
| 12 | 2,856×10⁻⁶ | 0,000000 % |

## VERDICT : ≤ 0,1 % → **pré-échelle 2^k_act CONFIRMÉE**

Fraction restante nulle sur toute la grille, y compris à `k_act = 11` : un
simple décalage binaire (`act/s_down × 2¹¹` avant `nvfp4_quant_act`,
`÷2¹¹` après déquantification côté sortie) suffit à faire disparaître
entièrement les 23-31 % de blocs perdus mesurés sans décalage — aucun bloc
n'était réellement à une magnitude physiquement nulle, ils étaient
seulement mal cadrés pour l'échelle E4M3 sans compensation globale. Pas
besoin du repli « amax par appel propre à down seul » (plus coûteux, une
vraie échelle flottante par appel).

## Conséquence

Correctif bon marché disponible pour poste4 : décalage binaire fixe
`2^k_act` (recalculé par appel ou par couche, coût trivial —
`math.floor(log2(672/max_amax))`, un seul `amax` global déjà nécessaire au
calcul actuel) sur l'entrée de `nvfp4_quant_act` pour `down_proj`
uniquement (`model.py:1028`), inversé après le GEMM. Équivalence
pile/boucle avec la table réelle + correctif dans le même commit, re-PPL
scellé ≤ 1,010 (ordre poste7, `poste7-glm-mma0-verdict` §3). poste2 attend.
