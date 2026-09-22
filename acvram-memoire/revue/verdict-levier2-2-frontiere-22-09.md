# Levier 2 (2) : frontiere-pas A(flux)/B(épinglé) — RÉFUTÉ sur trou_gpu, gain énorme mais ailleurs — 22/09 (poste2)

* instrument : `outils/gpu/mesure/frontiere-pas.py` (worktree poste2-w-21-09, main à jour), 4 processus A0/B0/B1/A1, Coder nvfp4, b=12, 300 pas pleins
* commit : main à jour (levier 2 fusionné)
* régime : A = `rapatriement=flux` (défaut, sampler=graphe) ; B = `rapatriement=epingle` (`ACVRAM_RAPATRIEMENT_EPINGLE=1`), confirmé sur la ligne de régime des 4 runs
* scellé (poste1, `poste1-levier-2-conception-22-09`) : tenu ssi `trou_gpu` baisse à 10-30 µs (contre 173 µs mesuré au levier 1) ; réfuté si `trou_gpu` reste > 100 µs — prédit aussi `consommer` ≤ 60 µs et `attente_evt` ≈ 6 500 µs
* mesuré : A `trou_gpu` 166,8/163,1 µs (moy 165,0), `consommer` 6702,0/6719,1 µs (moy 6710,6), `attente_evt` 5,6/5,2 µs ; B `trou_gpu` 186,9/176,6 µs (moy 181,8), `consommer` 15,7/15,1 µs (moy 15,4), `attente_evt` 5,6/5,3 µs. `echantillon` inchangé (3,7 µs, déjà optimisé par le levier 1)
* verdict : **RÉFUTÉ sur le critère prédit** — `trou_gpu` NE baisse PAS (165,0 → 181,8 µs, légèrement PIRE, > 100 µs de seuil de réfutation) : la carte reste oisive le même temps, ce n'est PAS elle que le rapatriement épinglé raccourcit. **Mais gain massif ailleurs, non prédit sous cette forme** : `consommer` (le D2H `.tolist()`) chute de 6710,6 → 15,4 µs (×436), cohérent avec la théorie de la mémoire épinglée (transfert asynchrone au lieu d'un D2H bloquant) — mais ce coût était côté HÔTE, hors `trou_gpu`, et n'était visiblement pas le facteur limitant du débit selon cette frontière. `attente_evt` inchangé (pas la signature ≈6500µs prédite).
* durée : 4 passes, quelques minutes (comparable au levier 1)

## Suite
Le gain ×436 sur `consommer` ne garantit pas de gain en débit servi puisque `trou_gpu` (la carte oisive, la vraie frontière) ne bouge pas — l'ABBA tranchera si le gain hôte se répercute quand même (recouvrement pipeline) ou s'il est absorbé sans effet sur le t/s. Cause à nommer par poste1 : pourquoi le rapatriement épinglé ne réduit-il pas `trou_gpu` comme prédit ?
