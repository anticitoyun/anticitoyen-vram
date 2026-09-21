# Sampler vectorisé b=12, 6f v2 (garde intra-fenêtre) — TENU/RÉFUTÉ — 21/09

* instrument : `scratchpad/poste4-b12-21-09/chaine-sampler-6f-v2.sh` (nouveau, garde corrigée : 3 échantillons `ps` pendant chaque fenêtre au lieu d'un load1 qui se refermait sur son propre harnais)
* commit : d12b2a08, worktree figé `poste2-d12b2a08-21-09`, arbres A/B chauds (chauffe à blanc 15s/15s)
* régime : charge étrangère = somme %CPU hors {acvram serve, harnais, Xorg/cinnamon, ps} ; fenêtre rejetée si un échantillon > 50 % ou watts moyen > 395 ; fenêtres enchaînées sans pause
* scellé : médiane(B) ≥ médiane(A)×1,02
* mesuré : A1 1559,7 (**etr_max 571,5 % — REJETÉE**, pic transitoire au 1er échantillon, cause non identifiée, 2 échantillons suivants à 30,1 %) · A2 1541,9 (etr_max 30,1) · A3 1538,9 (etr_max 30,0) · B1 1506,2 (etr_max 30,2) · B2 1471,9 (etr_max 30,0) · B3 1522,5 (etr_max 29,9). Watts : A 385,7-387,8, B 385,1-385,8 — tous sous 395, aucun rejet puissance.
* verdict : **RÉFUTÉ** — A retenues = [1541,9 ; 1538,9], médiane 1540,4 ; B = [1506,2 ; 1471,9 ; 1522,5], médiane 1506,2 ; ratio B/A = 1506,2/1540,4 = **0,978** (< 1,02). Le calcul automatique du script (qui incluait à tort A1) donnait déjà RÉFUTÉ (ratio 0,977) — même conclusion en excluant proprement la fenêtre polluée. Sur 2 mesures propres du bras A et 3 du bras B, le sampler vectorisé ne montre pas le gain de 2 % attendu ; A1 publié isolément (571,5 % étranger), pas dans la cellule.
* durée : 405 s (15:37:14–15:44:00), 6/6 fenêtres mesurées, 5/6 retenues

nvidia-smi propre avant/après.

## Suite
A1 rejetée sans cause identifiée (pic ponctuel non retrouvé sur les 2 échantillons suivants de la même fenêtre) — pas d'action requise, juste écarté du calcul comme prévu par le protocole.
