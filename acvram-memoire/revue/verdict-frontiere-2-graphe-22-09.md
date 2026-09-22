# Frontière (2) : frontiere-pas.py A B B A sous ACVRAM_SAMPLER_GRAPHE — TENU — 22/09 (poste2)

* instrument : `outils/gpu/mesure/frontiere-pas.py` (worktree poste2-w-21-09, main à jour), 4 processus séquentiels A0/B0/B1/A1, Coder nvfp4 défaut, b=12, ctx 2048, 300 pas pleins chacun
* commit : b0c25910 (fast-forward avant la prise)
* régime : A = `sampler=lent` (défaut) ; B = `sampler=graphe` (`ACVRAM_SAMPLER_GRAPHE=1`), confirmé sur la ligne de régime des 4 runs ; graphes on(hybrides≤12), pipeline=1
* scellé (poste1, `poste1-levier-1-conception-21-09` § 5) : tenu ssi `echantillon + trou_gpu` (médiane, 300 pas) baisse ≥ 30 µs entre A et B ; réfuté si < 30 µs ; alarme si gain > 1,4 % du pas (mesurerait autre chose)
* mesuré : A0 41,2+188,9=230,1 µs · A1 42,2+188,9=231,1 µs (moy. A=230,6) ; B0 3,7+172,3=176,0 µs · B1 3,7+173,6=177,3 µs (moy. B=176,65) — **baisse 53,95 µs**. `echantillon` seul chute de 41,7 → 3,7 µs (noyaux capturés, plus de lancements hôte) ; `trou_gpu` baisse aussi 188,9 → 172,9 µs (hôte moins limitant)
* verdict : **TENU, nettement** (53,95 µs ≥ 30 µs seuil) — 0,79 % du pas total (~6 800 µs), dans la fourchette prédite « trou_gpu ≥ 150 µs → +0,6 à +1,0 % » (trou_gpu mesuré 173-189 µs, régime hôte-limitant confirmé), sous le seuil d'alarme 1,4 %. A0≈A1 et B0≈B1 (paires cohérentes, pas de dérive de poste).
* durée : 71 s pour les 4 passes (02:53:17-02:54:28), bien sous les ≤ 3 min/passe prévues

## Suite
ABBA débit lent/graphe pour confirmer sur le t/s servi (au lieu de lent/lot), puis scellé E.
