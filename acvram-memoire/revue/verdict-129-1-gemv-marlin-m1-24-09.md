# Verdict — 129 (1) : GEMV sur disposition Marlin (E = 1) contre GEMV naturel à M = 1, Qwen3.8-27B — 24/09 02 h 4x (poste1)

* **instrument** : `scratchpad/poste1-p129-24-09/banc-gemv-marlin-m1.py` (formes du manifeste, poids NVFP4 aléatoires, L2 froid, un graphe, médiane de 20 ; down en deux moitiés de K, K > 11 264 ; `+gcol` pour les poids empilés) ; prise `prise-banc.sh`
* **commit** : 1e23cf9e (poste1-mtp)
* **régime** : RTX 5090, -lgc 2700 posé et rendu, cpu-safe=off (100/100), compute-apps début = fin (llama-server sur la 3080 Ti)
* **scellé** : `scratchpad/poste1-p129-24-09/scelle.md` (avant la mesure) — FAUX si la projection b=1 perd plus de 3 %
* **mesuré** (µs par appel, naturel → Marlin) : gate‖up 63,83 → 74,54 avec gcol (**+16,8 %**) ; down 36,73 → 43,06
  (**+17,2 %**, moitiés de K) ; GDN qkv 21,79 → 23,34 (+7 %) ; GDN gate 14,67 → 15,30 (+4 %) ; GDN out et o_proj
  13,58 → 16,95 (**+25 %**) ; q 25,44 → 28,21 (+11 %) ; tête 526,7 → 568,2 (+8 %). Justesse identique dans les deux
  bras (même erreur relative à 10⁻⁵ : le découpage de K est juste). **Projection b=1 : 10,13 → 11,65 ms, +15,0 %.**
* **verdict** : **FAUX** (+15,0 % > 3 %). Ma prédiction (+2 à +5 %) était trop optimiste : le GEMV Marlin perd aussi sur
  les grandes formes (+17 % sur gate‖up, là où j'attendais ± 5 %). La disposition UNIQUE partout coûterait ≈ 1,5 ms
  par pas à b=1.
* **durée** : 02:37:23 → 02:37:25 (banc) ; prise < 1 min

## Arbitrage par forme (gain b=8 au banc 128/Marlin, perte b=1 ici, mémoire d'une double copie)

| forme | gain b=8 | perte b=1 | double |
|---|---|---|---|
| gate‖up | 3,43 ms | 0,70 ms | 6,52 Go |
| down | 2,01 | 0,41 | 3,26 |
| GDN out | 0,46 | 0,16 | 0,85 |
| GDN qkv | 0,62 | 0,07 | 1,42 |
| o_proj | 0,16 | 0,06 | 0,30 |
| q_proj | 0,38 | 0,05 | 0,60 |
| tête | 0,48 | 0,04 | 0,72 |
| GDN gate | 0,47 | 0,03 | 0,85 |

Doubler (garder le naturel pour M = 1) dans l'ordre de perte : gate‖up → perte b=1 8,1 % (+6,5 Go) ; + down → 4,1 %
(+9,8 Go) ; **+ GDN out → 2,5 % (+10,6 Go)** : c'est le premier point sous 3 %. Les autres formes passent en Marlin seul.

## Options (au chef)
* **(A) Mixte** : deux dispositions pour gate‖up, down et GDN out (+10,6 Go), Marlin seul ailleurs → b=1 ≈ −2,5 %
  sur les GEMV, b=8 ≈ −7,9 ms. Poids ≈ 27,1 Go sur 32 : à prouver au chargement (le préfill long et le KV restent à
  loger), sinon OOM nommé.
* **(B) Rendre le GEMV Marlin aussi rapide que le naturel** (noyau : il perd 17 % sur des formes où les octets sont
  les mêmes, donc c'est l'accès, pas la bande), puis disposition unique (0 Go en plus). C'est du code de noyau, au
  résultat incertain.
* (C) Disposition unique, en acceptant −15 % sur les GEMV à b=1 : refusé par le critère du scellé.
