# Sampler vectorisé b=12, 6 fenêtres A/B, garde load1/écran ≤ 1,5 — 21/09

* instrument : `scratchpad/laurine-b12-21-09/chaine-sampler-6f.sh` (nouveau, fichier suivi)
* commit : d12b2a08, worktree figé `manon-d12b2a08-21-09`
* régime : chauffe à blanc A et B hors mesure, contrôle < 120 s ; garde load1-écran ≤ 1,5 avant chaque fenêtre
* scellé : 6 fenêtres A/B, médiane(B) ≥ médiane(A)×1,02 ET ≥ 1596 t/s ET puissance < 400 W
* mesuré : chauffe A0 15 s, B0 14 s (arbres déjà chauds — confirme que le 900 s vu précédemment était bien un JIT froid, pas un blocage) ; **0/6 fenêtres de mesure ouvertes** — load1 1,91, écran 0,2, load1−écran = 1,71 > 1,5 sur toute la tentative (15:33:05–15:33:06)
* verdict : **INDÉCIDABLE, garde de charge respectée** — aucune fenêtre n'a été mesurée sale ; la charge ambiante était trop élevée du début à la fin de cette tentative (~1 minute), pas de fenêtre calme trouvée. Bug corrigé avant cette prise : `passe()` retournait le code de `wait` (process tué par le script lui-même) au lieu de refléter le vrai succès, faisant échouer le `&&` même après une chauffe réussie — `return 0` explicite ajouté.
* durée : 32 s de carte (2 chauffes), tentative totale 32 s, aucune mesure prise

nvidia-smi propre avant/après.

## Suite
Retenter dans une fenêtre plus calme (load1-écran ≤ 1,5) ; les arbres A et B sont maintenant chauds (chargement < 20 s), donc une prochaine tentative sera rapide dès que la garde passe.
