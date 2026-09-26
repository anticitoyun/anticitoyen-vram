# Verdict — pièce 205 : table canal étendue aux formes int8 du Coder-30B — gain 0,094 ms/pas < seuil 0,1 : les formes du Coder RESTENT HORS TABLE, pas de KL ni d'ABBA (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p205-25-09/banc-canal-205.py` (= banc 195 : 54 géométries × 2 formes, poids int8 par canal
  synthétiques, L2 froid ≥ 256 Mo en rotation, graphe, médiane/40, M = 8 ; référence servi = vue g128 `gemm_etroit(compact)`) ;
  résultat `banc-canal-resultat.txt`.
* **commit** : 4878bb876 (poste6-205 = origin/main 0a785bd50 + scellé + banc) ; instruments KL/ABBA (4f4f79a4a) écrits, NON joués.
* **régime** : carte 0, horloge libre (2 865-2 955 MHz), compute-apps début = fin ; prise poste6-p205-banc après 701 s de file
  (une 1re prise abandonnée à 1 800 s : `ACVRAM_ATTENTE` relevée à 5 400).
* **scellé** : `scelle-etape1.md` (4878bb876, avant) : seuil total ≥ 0,1 ms/pas (chef) et ≥ 0,05 par forme ; prédit 0,28-0,33.
* **mesuré** (µs/appel, 48 appels/pas chacune) : qkv 5120 × 2048 : servi 9,48 → canal 8,23 (BN 32, BK 128, 4 w, 4 ét., 160 prog.,
  82 % du plancher) = **0,060 ms/pas** ; o 2048 × 4096 : servi 9,83 → canal 9,13 (BN 16, BK 256, 128 prog., 59 %) = **0,034** ;
  total **0,094 ms/pas** (1,9 % du pas b=8 de 4,996 ms).
* **verdict** : **sous le seuil** (0,094 < 0,1 ; o sous 0,05) → falsificateur « total < 0,1 → pièce close sans code servi »
  déclenché ; prédiction 0,28-0,33 FAUSSE, l'issue gênante nommée au scellé (K court, rampe non amortie, peu de programmes) est
  ce qui s'est produit, en pire sur o. Table inchangée ; ni KL ni ABBA (ils auraient coûté 40 min de carte pour 1,9 %).
* **durée** : prévu ≤ 3 min ; tenu 20 s de carte, 42 min de file ; à sec 25 min.

## Pourquoi le K entier ne gagne presque rien ici
Sur le mixte (195) les formes ont K 5 120-17 408 et N 5 120-34 816 : 160-544 programmes à K entier, chacun 20-70 Kio de poids par
tuile, et le noyau à tranches perdait 15-20 % dans ses partiels. Ici N = 2 048 (o) donne 64 programmes à BN 32, 128 à BN 16 —
la carte a 170 SM : le noyau est limité par les programmes (59 %), et le noyau à tranches servi (`[32×11]`, 352 programmes) fait
déjà 55 %. qkv (N 5 120, K 2 048) : 160 programmes, 4 itérations de BK 512 ou 16 de BK 128 — la rampe d'un appel de 8 µs n'est
pas amortie (82 %). Le levier restant sur ces formes est le plancher des petits appels (193 § 2 : ≈ 2,9 µs), pas la géométrie ;
une grille 2-D à K entier par sous-tranche avec réduction atomique serait hors bit ET non déterministe — je ne la propose pas.

## Suite (à chef)
Rien à servir. Les 0,96 ms/pas d'int8 étroit du Coder à b=8 (19 % du pas, 203) restent sur le noyau à tranches ; le gain réel y
est de 0,06-0,09 ms au mieux avec l'outillage actuel. Si l'attention du Coder doit gagner, c'est par le format (nvfp4 des
projections, 99/42 : −0,119 J/pas sur le mixte) ou par les trous entre noyaux (203 poste 2), pas par ce noyau.
