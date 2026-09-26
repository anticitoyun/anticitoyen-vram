# 221 — GEMV int8 en UN lancement, tranche en indice de grille rapide (poste1, 26/09) — AU BIT, mais PLUS LENT : FAUX

* instrument : `scratchpad/poste1-p221-25-09/` — `prise1.sh` (compilation, cuobjdump, tests 221 + 187, deux bras cassants en
  copie jetable, `banc.py` ABBA en sous-processus, L2 vidé par fenêtre) ; `prise2.sh` (ABBA servi) ARRÊTÉE après la passe A1
* commit : poste1-221 8b8c86049 (code 87ed0da69 : noyau `ntr`, témoin `ACVRAM_INT8_GEMV_BOUCLE=1`)
* régime : carte 0, -lgc 2700 (banc), cpu-safe 100 ; poids réels du mixte (couche 0 : qkv 10240 × 5120, porte 6144 × 5120,
  sortie 5120 × 6144), tranche 6/6
* scellé : `scratchpad/poste1-p221-25-09/scelle.md` (f086daa98 + addenda 1-2, avant chaque prise)
* mesuré : registres 128 (≤ 128 tenu) ; tests 12/12 verts ; cassants décalage et garde ROUGES ; banc : fusion +19 à +25 % à
  n = 78-80, +10 à +29 % à n = 8-16, 0 à +2 % à n ≤ 6
* verdict : **au bit prouvé, levier FAUX** (prédit −60 à −85 % à n = 78 ; seuil −30 %) — l'issue gênante, en pire
* durée : prise 1 02:51:41-02:57:53 (journal du verrou, tenue 372 s ; l'essai de 02:28:51 s'était arrêté sur cuobjdump),
  prise 2 03:15:15-03:18:18 (183 s, arrêtée)
* contamination (chef, 26/09) : le pytest orphelin d'poste6 (03:13-03:23, hors verrou) ne touche PAS les chiffres du
  verdict — tous viennent de la prise 1 (banc en fin de prise, avant 02:57:53) ; load1 de la prise 1 : médiane 2,0, max 3,5
  (compilation et tests de la même prise). Seule la passe A1 de la prise 2, non retenue, tombe dans la fenêtre.

## Banc isolé (µs par appel, moyenne de A1/A2 et de B1/B2, écarts intra-bras ≤ 0,5 %)
| n | qkv boucle → fusion | porte | sortie |
|---|---|---|---|
| 2 | 24,1 → 24,1 (0 %) | 16,1 → 16,4 | 15,2 → 15,5 |
| 6 | 50,9 → 50,9 (0 %) | 33,2 → 33,3 | 29,2 → 29,6 |
| 8 | 74,3 → 96,1 (**+29 %**) | 49,0 → 61,0 (+25 %) | 43,6 → 56,3 (+29 %) |
| 12 | 101,2 → 104,2 (+3 %) | 66,3 → 66,5 | 58,1 → 63,2 (+9 %) |
| 16 | 137,2 → 154,4 (+13 %) | 89,8 → 98,8 (+10 %) | 79,1 → 93,7 (+18 %) |
| 78 | 654,9 → 781,4 (**+19 %**) | 428,8 → 486,3 (+13 %) | 377,3 → 470,0 (+25 %) |
| 80 | 678,7 → 849,1 (+25 %) | 444,5 → 513,2 (+15 %) | 393,4 → 490,3 (+25 %) |

## Ce que ça réfute (ERRATUM de la 204 b, § Mécanisme)
La 204 b disait « chaque lancement relit tout W **en DRAM**, 13 lectures ». Faux : la boucle séquentielle relit W depuis
le L2 (qkv 52 Mo), et le GEMV à n = 78 n'est pas borné par la DRAM. À n = 8, la fusion calcule 12 lanes pour 8 (bourrage
de la 2e tranche) et perd 29 % : le noyau est borné par le CALCUL (FMA sur cœurs CUDA, déquantification par poids et par
lane), pas par les octets. À n = 78, sans bourrage (13 × 6), elle perd encore 19 % : entrelacer les tranches ne rend
rien et coûte (cause non mesurée ; hypothèses : plus de blocs résidents qui se disputent x et le L2, ordre de lancement).
Les 64,6 ms/séq de la 204 b sont donc du calcul sur cœurs CUDA : le seul levier restant est un GEMM sur cœurs tenseurs,
**hors bit** (la 179 de poste5 : seuil GEMV/GEMM int8 abaissé sous B', jugé par KL), ou la sortie de la boucle par séquence
(LOT=1, refusé en qualité à la 165).

## Suite
* Code NON fusionné ; branche poste1-221 gardée (témoin, test memcheck et test au bit réutilisables).
* Proposition (décision chef) : relancer la 179 b (seuil GEMV/GEMM int8 sous B', KL contre témoins) avec la 204 b comme
  base (préfill 8 × 78 : 886 ms, dont 578 d'int8_gemv).
* Prise 2, passe A1 seule, non retenue : b=16 668,4 t/s, TTFT C1 médiane 940,8 ms.
