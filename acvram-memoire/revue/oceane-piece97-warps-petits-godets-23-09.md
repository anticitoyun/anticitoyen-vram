# Pièce 97 — 4 warps aux godets ≤ 4 : PAS AU BIT (le test l'a dit) ; défaut retiré — 23/09 (Océane)

* **instrument** : `tests/test_glue_compact.py::test_warps_petits_godets_au_bit` : 8 warps contre 4, avec `torch.equal`, cellules b ∈ {1, 2, 4} × ctx ∈ {768, 2 048} plus (3, 40). Le lot mêle les longueurs (ctx − 7i). Le bras BN 16 doit casser. S'y ajoutent `capture-godets.py` aux godets 1/2/4 (Coder qkvo-i8c, gemma-4-26B-A4B) et les tests d'attention, de glue et de graphes.
* **commit** : b575dfff (défaut 4 aux godets ≤ 4, version 0.6.39, ligne gelée) → retiré par ba39adc1, qui revient à 0.6.38 et au défaut 8. b575dfff n'a été poussé que sur `oceane-alpha-experts`, **jamais sur main**.
* **régime** : prise de 16:11:02 à 16:12:08 ; au début et à la fin, seul llama-server 4627 tourne ; détail des tests relu à 16:1x (2 min, verrou).
* **scellé** : l'ordre du chef, « test au bit : il doit casser si l'on change l'ordre de réduction ».
* **mesuré** :
  * test : **5/7 au bit**. (2, 768) et (4, 768) diffèrent de 1 ulp bf16 (max |Δ| 0,0039 et 0,0020). Les tests d'attention, de glue et de graphes passent (36 passés, 3 ignorés, hors ces 2 échecs).
  * capture : 3/3 godets sur chacun des 2 alias (Coder 2,98 / 4,10 / 4,61 ms ; gemma 4,21 / 5,36 / 6,38 ms), ligne `glue=compact(8,petits=4)`.
* **verdict** : **faux au critère « au bit »**.
  1. Le « au bit 6/6 » du banc de la pièce 96 était un effet de régime : le banc donnait la même longueur à toutes les séquences. Dès que les longueurs sont mêlées, les réductions croisées entre warps (`tl.sum` de p et de l·w) changent d'ordre.
  2. Le code l'écrivait déjà (commentaire C15-3c, `attn_paginee.py` : « ≤ 1 ulp 16 bits sur carte »), tout comme la pièce 92 (4 warps au bit sur 10/11 cellules seulement).
  3. Le test a fait son travail. Le défaut est retiré, et je n'ai pas prévenu Manon : son ABAB n'était ordonné que si le critère tenait.
* **ce qui reste possible (au chef)** : 4 warps aux godets ≤ 4 relève désormais du critère ulp + KL, comme la 82 ter, et non plus du critère au bit.
  * Gain au banc : −0,7 à −1,5 µs/couche, soit ≈ 1-2 % du pas à b=1.
  * Deux voies : le passer au défaut sous le critère KL (b=1 identique à ±0,005, b=2/4 ≤ témoin + 0,025), ou le garder en opt-in « rapide ± 1 ulp » (REGLES § 1).
* **LEÇON** : un « au bit » de banc à longueurs égales ne prouve rien pour un lot mêlé. La cellule du banc doit porter le lot mêlé, comme le test.
* **durée** : ≈ 35 min ; carte ≈ 1 min 06 + 1 min.

## Addendum 16 h 2x — critère préexistant (ordre du chef) : FAUX, abandon définitif

* **instrument** :
  * `scratchpad/oceane-p97-23-09/ulp-bras.py` : un bras par processus. Coder qkvo-i8c servi par Engine, graphes demandés, glouton, 32 jetons, lot mêlé (invites de 740/712/761/689 jetons) à B=4 et 740 à B=1 ; logits de chaque pas relevés.
  * `ulp-compare.py` : `verdict_decodage` (`acvram/quant/equivalence.py:162`).
  * `kl-b.py` : celui de la p81/p82, avec des séquences j > 0 raccourcies de 7·j (lot mêlé) et un compte des appels.
* **commit** : af6ef6f0 (opt-in `ACVRAM_ATTN_WARPS_PETITS`, défaut 8 inchangé).
  * **régime** : éco 2 700 ; au début et à la fin, seul llama-server 4627 tourne ; prise de 16:17 à 16:19:37.
  * **scellé** : `scelle-critere.md`, commité et poussé avant la prise.
* **mesuré** :

| contrôle | témoin | 4 warps | critère | |
|---|---|---|---|---|
| test sur carte (lot mêlé, ≤ 1 ulp de la valeur, BN 16 casse) | — | 5/5 | 5/5 | tenu |
| ulp b=1 (33 positions) : med · p90 · max | 0 · 0 · 0 (graphes = eager au bit) | 2,92 · 6,48 · 10,25 | ≤ 1,2 × témoin, max ≤ max(T) | **FAUX** |
| ulp b=4 (130 positions) | 0 · 0 · 0 | 3,48 · 9,91 · 52,24 ; 0 divergence top-1 | idem | **FAUX** |
| KL b=1, kl_max par invite | 0,0073 · 0,1185 · 0,5193 · 0,0676 · 0,2326 | 0,0073 · 0,1185 · 0,5193 · **0,0544** · 0,2326 | ≤ 0,74 et ≤ T + 0,05 | tenu |
| KL b=4 | 0,0078 · 0,0972 · **0,8103** · 0,0676 · 0,302 | identique au témoin | ≤ 0,74 sur 5/5 | **faux pour les DEUX bras** (invite 2) |

  Preuve que le chemin est pris : 144 (ulp) et 1 680 (KL) appels `paged_attention(compact)` par bras, avec les warps effectifs 8 ou 4 selon le bras.
* **verdict** : **FAUX, abandon définitif.** L'opt-in est retiré (commit 9570e4d6) ; les instruments et leurs sorties sont gardés, les logits en sha256.
  1. **Ma prédiction « b=1 au bit » est fausse.** Au service, les contextes ne sont pas des multiples de 64 et les tuiles partielles changent l'ordre des réductions croisées, même à b=1.
  2. **Le critère du 15/09 se réduit ici à « au bit ».** Le témoin graphes/eager vaut 0 ulp : c'est l'invariant du dépôt, prévu et dit au scellé. Aucune optimisation qui touche l'ordre de sommation ne peut donc le passer sur ce chemin.
  3. **Fait nouveau, hors 97.** Le DÉFAUT, 8 warps, dépasse le seuil KL de 0,74 sur l'invite 2 à b=4 en lot mêlé : 0,8103, alors que b=1 donne 0,5193 et que la p82 ter donnait 0,410 à b=12 avec des invites identiques. La composition du lot change donc la sortie de la séquence 0. C'est attendu à l'ordre près (godets, tensor MoE aux godets ≥ 8 non concernés à b=4), mais c'est au-dessus du seuil. À signaler, et à rejouer par Manon ou par moi sur ordre, avant qu'une KL à lot mêlé ne serve de porte.
* **durée** : prise de 2 min 35 ; environ 1 h en tout.
