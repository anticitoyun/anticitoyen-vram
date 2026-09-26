# familles-noyaux --detail proj_etroites_int8 — TENU, q/o ≈ 10-12 µs — 22/09 (poste2, à sec)

* instrument : `outils/gpu/mesure/familles-noyaux.py --detail proj_etroites_int8` sur la trace nsys déjà écrite (`scratchpad/nsys-familles-22-09-v3/graphe_cuda_gpu_trace.csv`), 0 min de carte
* commit : main à jour (a1673de0+)
* régime : celui de la trace du 22/09 (leviers 1+2 posés, b=12)
* scellé (poste1) : q et o ≈ 10 µs chacun (0,84 To/s) ; réfuté si q/o ≥ 1,3 To/s (temps sensiblement plus court)
* mesuré : `_etroit_reduit_kernel [32x11]` 11,8 µs × 48 lancements/pas (0,564 ms) ; `_etroit_reduit_kernel [80x4]` 10,3 µs × 48 (0,495 ms) ; `[2374x1]` 206,2 µs × 1 (hors régime établi, exclu) ; `splitKreduce [4x1]` 1,0 µs × 48
* verdict : **TENU** — deux noyaux dominants à 10,3 et 11,8 µs, cohérents avec ~0,84 To/s prédit ; aucun ne s'approche du seuil de réfutation 1,3 To/s (qui impliquerait ≈ 7,7 µs). Attribution q vs o non faite ici (grilles `[32x11]`/`[80x4]` non nommées explicitement dans la sortie) — à confirmer par poste1 si l'attribution précise importe.
* durée : 0 min de carte (à sec)

## Suite
PPL relative A/B 31B ensuite.
