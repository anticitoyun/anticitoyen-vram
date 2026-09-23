# banc-etroites-noyaux (Océane) — RÉFUTÉ sur (b), (c) crash non mesuré — 22/09 (Manon)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/banc-etroites-noyaux.py --rep 200`, sha 9f57b402
* scellé (Océane) : (c) FP8 `_scaled_mm` 1,3-1,8 To/s ; (b) `_int_mm` 1,13 qkv / 0,67 o ; réfuté si aucun >1,0 To/s ; alarme >2,0
* mesuré :

| noyau | qkv | o | au bit |
|---|---|---|---|
| (a) triton int8 groupe (défaut actuel) | 1,181 To/s (76% plancher) | 0,897 To/s (58%) | True |
| (b) int_mm int8 canal | 0,933 To/s (60%) | 0,552 To/s (36%) | **False (écart 0,012/0,015)** |
| (c) scaled_mm fp8 | **crash** : `RuntimeError: Expected b.stride(0) == 1 ...` | même crash | — |

* verdict : **RÉFUTÉ sur (b)** (sortie de l'instrument) — plus lent que le défaut ET pas au bit (écart 0,01-0,015, disqualifiant même sans le débit). **(c) non mesuré** — crash de forme/stride avant tout chronométrage, ni confirmé ni réfuté sur sa prédiction 1,3-1,8 To/s. Le défaut actuel (a, triton int8 groupe) reste le meilleur candidat mesuré, dépassant même 1,0 To/s sur qkv (1,181) mais pas sur o (0,897).
* durée : <1 min de carte

## Suite
(c) `_scaled_mm` FP8 à corriger avant de pouvoir trancher (erreur de stride sur l'argument `b`, probablement une transposition manquante avant l'appel) — à Océane si elle veut ce noyau testé.
