# Pièce 97 — 4 warps aux godets ≤ 4 : PAS AU BIT (le test l'a dit) ; défaut retiré — 23/09 (poste1)

* **instrument** : `tests/test_glue_compact.py::test_warps_petits_godets_au_bit` : 8 warps contre 4, avec `torch.equal`, cellules b ∈ {1, 2, 4} × ctx ∈ {768, 2 048} plus (3, 40). Le lot mêle les longueurs (ctx − 7i). Le bras BN 16 doit casser. S'y ajoutent `capture-godets.py` aux godets 1/2/4 (Coder qkvo-i8c, gemma-4-26B-A4B) et les tests d'attention, de glue et de graphes.
* **commit** : b575dfff (défaut 4 aux godets ≤ 4, version 0.6.39, ligne gelée) → retiré par ba39adc1, qui revient à 0.6.38 et au défaut 8. b575dfff n'a été poussé que sur `poste1-alpha-experts`, **jamais sur main**.
* **régime** : prise de 16:11:02 à 16:12:08 ; au début et à la fin, seul llama-server 4627 tourne ; détail des tests relu à 16:1x (2 min, verrou).
* **scellé** : l'ordre du chef, « test au bit : il doit casser si l'on change l'ordre de réduction ».
* **mesuré** :
  * test : **5/7 au bit**. (2, 768) et (4, 768) diffèrent de 1 ulp bf16 (max |Δ| 0,0039 et 0,0020). Les tests d'attention, de glue et de graphes passent (36 passés, 3 ignorés, hors ces 2 échecs).
  * capture : 3/3 godets sur chacun des 2 alias (Coder 2,98 / 4,10 / 4,61 ms ; gemma 4,21 / 5,36 / 6,38 ms), ligne `glue=compact(8,petits=4)`.
* **verdict** : **faux au critère « au bit »**.
  1. Le « au bit 6/6 » du banc de la pièce 96 était un effet de régime : le banc donnait la même longueur à toutes les séquences. Dès que les longueurs sont mêlées, les réductions croisées entre warps (`tl.sum` de p et de l·w) changent d'ordre.
  2. Le code l'écrivait déjà (commentaire C15-3c, `attn_paginee.py` : « ≤ 1 ulp 16 bits sur carte »), tout comme la pièce 92 (4 warps au bit sur 10/11 cellules seulement).
  3. Le test a fait son travail. Le défaut est retiré, et je n'ai pas prévenu poste2 : son ABAB n'était ordonné que si le critère tenait.
* **ce qui reste possible (au chef)** : 4 warps aux godets ≤ 4 relève désormais du critère ulp + KL, comme la 82 ter, et non plus du critère au bit.
  * Gain au banc : −0,7 à −1,5 µs/couche, soit ≈ 1-2 % du pas à b=1.
  * Deux voies : le passer au défaut sous le critère KL (b=1 identique à ±0,005, b=2/4 ≤ témoin + 0,025), ou le garder en opt-in « rapide ± 1 ulp » (REGLES § 1).
* **LEÇON** : un « au bit » de banc à longueurs égales ne prouve rien pour un lot mêlé. La cellule du banc doit porter le lot mêlé, comme le test.
* **durée** : ≈ 35 min ; carte ≈ 1 min 06 + 1 min.
