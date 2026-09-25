# Scellé — pièce 183 : seuil GEMV int8 abaissé dans la portée de B' (poste5, 25/09, AVANT la prise)

Ordre de chef (registre 05 h 36) : « 183 : INT8_GEMV_MAX abaissé (sortie changée, scellé) ». Suite du verdict 179
(« Le vrai levier du banc ») : à L = 78, le préfill int8 du mixte prend le GEMV par tranches, 1 195 ms pour 8 séquences,
contre 439 ms en déquant partagée + GEMM à L = 92.

## Code
`kernels.int8_matmul` (kernels/__init__.py) : `ACVRAM_INT8_GEMV_MAX_PARTAGE` (0 = coupé, défaut inchangé). Dans la
portée de B' seulement (`_W_PARTAGES` non nul : boucle par séquence d'une couche à récurrence linéaire), seuil
= min(80, valeur). Hors portée (séquence seule, décodage, couches sans récurrence) : rien ne change. Variable dans
`regime.VARIABLES` et `cli.VARIABLES_LUES`. Valeur essayée : 16 (non optimisée ; limite).
**La sortie change** (GEMV et déquant + GEMM n'ont pas la même arithmétique) : opt-in, jugé par KL contre témoins.

## Tests (prise `tests`)
`tests/test_int8_seuil_partage_183.py` : hors portée au bit et GEMV ; dans la portée déquant, écart relatif
0 < rel ≤ 2⁻⁶ ; M = 12 reste GEMV. + 179, cadrage perplexité, régime. Bras cassant : condition de portée retirée
→ ROUGE (le test hors portée doit casser). **FAUX** si vert sous le cassant.

## KL (prise `kl`, `kl183.py`, critère de chef repris de la 165)
Par composition C1-C3 : KL max B ≤ 2 × max(T1, T2) ; argmax B ≥ T1 − 0,005 ; ΔPPL max B ≤ 2 × témoins ; rejeu A et B
à 0 ulp. T1 = séquences seules, T2 = ordre inverse.
* Qwen3.8-27B-unsloth-mixte-i8c : **prédit TENU** (écart ≤ ordre de l'arrondi bf16, sous l'écart des témoins).
* GLM-4.7-Flash-srcbf16-nvfp4 (MLA, pas de récurrence) : **prédit A = B au bit**, 0 déquant int8 dans B.
* Qwen3.8-27B-nvfp4 : **prédit A = B au bit** si `chemins_int8_B` ne compte aucune déquant ; sinon jugé au critère.
**FAUX** : un alias NON TENU ; un alias nvfp4 prédit au bit qui ne l'est pas sans déquant int8 comptée.

## Banc (prise `banc Qwen3.8-27B-unsloth-mixte-i8c`)
Banc chat de la 102, b=8, 256 jetons, -lgc 2700, fenêtre d'admission au défaut (179 b) ; A = seuil coupé (0),
B = 16 ; A B B A A B B A A B ; passe nulle si `repli_eager` ≠ 0 ou seuil absent de la ligne de régime en B.
**Prédit** : préfill ≈ 1,19 → ≈ 0,45 s par lot (1 pas de préfill par lot sous la fenêtre) ; lot ≈ 6,2 s
(8 × 256 à ≈ 330 t/s) → **B/A +8 à +14 %**. **FAUX** si B/A < +5 %. Au-dessus de +16 % : je mesure autre chose
(le préfill n'en vaut pas tant) et je le dis.

## Limites
Seuil 16 non balayé ; Qwen3.8 nvfp4 et GLM ne jugent que la portée ; pas de J/jeton scellé (relevé si le banc le rend).
