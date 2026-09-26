# banc-etroites-splitk, rejoué après 2 correctifs (poste1 e6c9ed8d, 2d0204ba) — RÉFUTÉ — 22/09 (poste2)

* instrument : `outils/gpu/mesure/banc-etroites-splitk.py --json`, sous carte.sh ; 1er essai crash (poids CPU, `to_device` absent, mon verdict précédent) ; 2e essai crash (garde device générique `cuda` ≠ `cuda:0`, signalé) ; **3e essai TENU** après les deux correctifs
* commit : main/poste2 à jour (5727a4b3 + merge 2d0204ba)
* scellé (poste1) : F 4-6 gagnant, qkv 10,3→≤7,5 µs, o 11,8→≤6,5 µs, ≤1 ulp sur <5 % ; réfuté si aucun F≤1 ulp ne gagne ≥2 µs ou si >1 ulp
* mesuré :

| forme | T auto | µs auto | T=2 (0 ulp) | µs T=2 | meilleur T≤1ulp | T avec ulp>1 |
|---|---|---|---|---|---|---|
| qkv [5120,2048] | 4 | 8,99 | 0 ulp | 9,09 (pire) | T=2, pas de gain | T≥6 (1 ulp), 11,3-13,0 µs, tous pires |
| o [2048,4096] | 11 | 10,14 | 0 ulp | 10,62 (pire) | T=2, pas de gain | T≥4 (2-3 ulp), 9,4-12,9 µs |

* verdict : **RÉFUTÉ** (sortie de l'instrument lui-même : « aucun T à ≤1 ulp ne gagne ≥2 µs — la marge n'est pas dans les tranches (occupation par programme : ptxas) »). Le seul T restant à 0-1 ulp (T=2) est plus LENT que le défaut auto sur les deux formes ; les T qui vont plus vite (o T=4, -0,76 µs) sortent à 2 ulp, hors règle « on ne livre pas ». Pas de levier ici : le split-K n'a pas de marge de temps de calcul, l'écart observé plus tôt (§1 de `poste1-ecart-trtllm-22-09.md`, plafond gagnable 0,45-0,70 ms) est ailleurs (probablement l'occupation SM/ptxas nommée par l'instrument, pas la stratégie de réduction).
* durée : <1 min de carte (2 essais crashés hors carte à 0 s, 3e mesuré)

## Suite
Levier étroites split-K fermé (réfuté au bit). ABBA « ± 1 ulp » annulé (pas de F gagnant à proposer). File : reconversion 30B-VL mediane_couche → (c)+P3 → KL acvram/bf16 → énergie 4 moteurs.
