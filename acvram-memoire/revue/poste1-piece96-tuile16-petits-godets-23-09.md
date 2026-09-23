# Pièce 96 — tuile BN 16 et 4 warps aux petits godets : FAUX (plus lente partout) ; à côté, 4 warps seuls, au bit, −0,7 à −1,5 µs — 23/09 (poste1)

* **instrument** : `outils/gpu/mesure/banc-attn-unifie.py acvram`, avec les variantes `ACVRAM_BANC_VARIANTES=servi,bn16w4,bn16w4-Crecalc,bn16w8,bn64w4` et les cellules b ∈ {1, 2, 4} × ctx ∈ {768, 2 048}. L2 froid, graphe, mur/48 ; CUPTI en information. Pour chaque variante : erreur contre fp64, écart à la sortie servie, et test au bit.
* **commit** : 23babcc2 (arbre = origin/main 732297c6 fusionné).
* **régime** : -lgc 2700, horloge moyenne sous charge 2 691 MHz ; au début et à la fin, seul llama-server 4627 tourne.
* **scellé** : `scratchpad/poste1-p96-23-09/scelle.md`, commité et poussé avant la prise. Variante principale : bn16w4 à C servi.
* **mesuré** (µs par couche, mur de graphe) :

| cellule | servi (BN 64, 8 w) | **bn16w4** (principale) | prédit | bn16w4-Crecalc | bn16w8 | bn64w4 (4 warps seuls) |
|---|---|---|---|---|---|---|
| b=1 ctx 768 | 7,82 | **9,63** | 5,5-7,0 | 12,57 | 10,37 | 7,16 (au bit) |
| b=1 ctx 2 048 | 13,95 | **15,82** | 10,0-12,5 | 15,84 | 16,50 | 12,54 (au bit) |
| b=2 ctx 768 | 8,54 | **10,10** | 6,0-7,5 | 16,72 | 10,82 | 7,38 (au bit) |
| b=2 ctx 2 048 | 16,21 | **18,39** | 11,5-14,5 | 18,29 | 19,12 | 14,70 (au bit) |
| b=4 ctx 768 | 11,25 | 14,75 | ≤ 11,55 | 17,27 | 15,12 | 9,93 (au bit) |
| b=4 ctx 2 048 | 15,43 | 20,42 | ≤ 15,73 | 21,48 | 20,59 | 14,53 (au bit) |

  * Justesse : toutes les variantes sont à 1,9-2,0e-3 contre fp64, et 100 % des lignes restent à ≤ 1,1 × la distance du servi.
  * Les variantes BN 16 s'écartent de 0,6 à 1,3 × 2⁻⁸ de la sortie servie ; bn64w4 est **au bit dans les 6 cellules sur 6**.
  * Répétabilité du servi contre la prise de 16:06 : 7,82 = 7,82 ; 13,95 contre 14,03 ; 8,54 contre 8,37 ; 16,21 contre 16,16.
* **verdict** :
  1. **FAUX.** bn16w4 est plus lente que le servi dans toutes les cellules, de +1,6 à +5,0 µs. Le seuil de faux était « > servi − 0,5 » ; il est franchi partout. Mes prédictions (−1,5 à −3,5) avaient le signe à l'envers.
  2. **L'issue nommée au scellé s'est produite.** bn16w8 est aussi plus lente, donc la tuile de 16 n'est pas notre levier. Le −28 % du noyau vLLM à b=1 vient d'ailleurs : réduction séparée plutôt que dernier arrivé et atomique, ou disposition des lectures. Non établi ; je ne le poursuis pas.
  3. **Trouvé à côté, non scellé : bn64w4**, c'est-à-dire notre tuile avec 4 warps au lieu de 8.
     * Il est au bit et plus rapide dans les six cellules : −0,66 (b=1, 768), −1,41 (b=1, 2 048), −1,16 (b=2, 768), −1,51 (b=2, 2 048), −1,32 (b=4, 768), −0,90 (b=4, 2 048) µs/couche.
     * Soit ≈ −0,03 à −0,07 ms/pas à b=1 (1 à 2 % du pas).
     * C'est cohérent avec la p92 (4 warps −15 % à b=1 ctx 2 048). Le défaut de 8 warps venait de l'ABAB servi à b=12 du 20/09 et n'a jamais été jugé aux petits godets.
* **ce que cela ordonne (au chef)** : fermer la 96. En pièce neuve, à sceller avant toute prise : `WARPS_COMPACT` = 4 pour les godets B ≤ 4, 8 au-delà.
  * Comme c'est au bit, pas besoin d'ulp ni de KL : il suffit du test au bit (test_glue_compact) et de la capture aux godets 1/2/4.
  * La cellule servie b=1 en ABAB reviendrait à poste2 (prédit −1 à −2 % du pas).
  * Coût ≈ 1 h de code et de test, 5 min de carte.
* **durée** : prévue 20 min de variante + 5 s de carte ; tenue ≈ 25 min, carte 8 s (16:08:31-39), verrou rendu.
