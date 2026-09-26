# Banc unified_attention aux petits lots — RÉFUTÉ au critère strict (3 cellules sur 4 ; b=1 ctx 2 048 manque de 0,15 µs) — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-attn-unifie.py`, cellules passées par `ACVRAM_BANC_CELLULES=1:768,1:2048,2:768,2:2048`. L2 froid, graphe, mur/48 ; somme des noyaux CUPTI en information.
* **commit** : cdb193b7. **régime** : -lgc 2700, horloge moyenne sous charge 2 692 MHz ; au début et à la fin, seul llama-server 4627 tourne ; confondu Triton 3.7.1 (b) contre 3.8.0 (c) nommé.
* **scellé** : `scratchpad/poste1-unifie-23-09/scelle-b1b2.md`, commité avant la prise. Critère : GO si (b) ≤ (c) − 1,5 µs dans les quatre cellules.
* **mesuré** (µs par couche ; Δ = (b) − (c)) :

| cellule | (a) fp8, info | (b) vLLM int8, notre cache | (c) acvram | Δ | ≤ −1,5 ? | prédit (b) / (c) |
|---|---|---|---|---|---|---|
| b=1 ctx 768 | 5,59 | 5,82 | 7,82 | **−2,00** | oui | 5,3-6,3 / 7,6-8,6 |
| b=1 ctx 2 048 | 11,42 | 12,68 | 14,03 | **−1,35** | **non** | 9-13 / 13,5-14,5 |
| b=2 ctx 768 | 6,67 | 6,56 | 8,37 | **−1,81** | oui | 5,5-7 / 7,8-9 |
| b=2 ctx 2 048 | 13,35 | 14,64 | 16,16 | **−1,52** | oui (de justesse) | 9-13 / 13,5-15,5 |

  * Répétabilité, b=1 ctx 768 contre la prise de 16:04 : (b) 5,82 = 5,82 ; (c) 7,82 contre 8,07 (−3,1 %). L'alarme de ± 5 % n'est pas déclenchée, mais la dispersion est d'environ 3 %, soit ≈ 0,4 µs à ctx 2 048.
  * Justesse contre fp64 : (b) de 3,1 à 4,2e-3, (c) de 1,9e-3. Aucun bras cassé.
* **verdict** : **RÉFUTÉ au critère strict.**
  1. b=1 ctx 2 048 manque le seuil de 0,15 µs, ce qui est moins que la dispersion du banc. Ce n'est pas un gain nul, c'est un gain sous le seuil.
  2. Lu par lot : b=2 passe aux deux contextes, b=1 passe à 768 seulement.
  3. L'issue que j'avais nommée s'est produite : l'écart se referme avec le contexte (−2,00 → −1,35 à b=1). Le noyau vLLM gagne par la latence aux contextes courts, pas par le débit.
  4. Le fp8 par tenseur de (a) est plus rapide que (b) de 1,3 µs à ctx 2 048 : l'échelle par jeton coûte, et le port garderait en plus le P·V en fp16.
  5. Gain à la clé si on portait quand même, limité aux godets ≤ 4 : −0,096 ms/pas à b=1 ctx 768 (≈ 3 % du pas) et −0,065 à ctx 2 048 (≈ 2 %).
* **durée** : écriture du scellé ≈ 10 min ; carte 1 prise de 5 s (16:06:22-27), verrou rendu.
