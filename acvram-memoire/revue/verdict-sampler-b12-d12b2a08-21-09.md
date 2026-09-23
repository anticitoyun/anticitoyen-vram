# Sampler vectorisé b=12, A/B dans la même fenêtre (LISEZ-MOI Laurine) — 21/09

* instrument : `scratchpad/laurine-b12-21-09/chaine-sampler.sh` (Laurine, copié depuis son worktree, exécuté tel quel)
* commit : d12b2a08, worktree figé `manon-d12b2a08-21-09`, `.venv` reconstruit
* régime : T1 (12 × 1024, invite 256, fenêtre 20 s), `-lgc 2700`, poste 1030 ; A = `ACVRAM_SAMPLER_LENT=1`, B = rien posé
* scellé : B ≥ A×1,02 ET B ≥ 1596 t/s → TENU ; B < A×1,02 OU B < 1570 → RÉFUTÉ ; alarme si A hors 1440-1480 → juger d'abord le ratio B/A par paire
* mesuré : A1 1459,2 · B1 1559,0 · A2 1511,5 · B2 1484,9 (t/s) ; moy A 1485,35, moy B 1521,95
* verdict : **INDÉCIDABLE, alarme dérive de poste déclenchée** — A2 (1511,5) est hors la fourchette 1440-1480 annoncée par le LISEZ-MOI (+3,6 % vs A1), le poste n'est pas resté stable pendant la fenêtre malgré l'alternance ABAB. Ratios par paire adjacente : B1/A1 = 1,068 (+6,8 %), B2/A2 = 0,982 (−1,8 %) — signes opposés, pas un effet net et stable du sampler vectorisé. Sur les moyennes brutes (à lire avec réserve vu l'alarme) : ratio global 1,025 (≥ 1,02, tenu de justesse) mais moy B = 1521,95 < 1570 → réfuté sur le seuil absolu. Équivalence NON JOUÉE : `scratchpad/laurine-b12-21-09/equiv-sampler.py` absent de l'arbre (annoncé par le LISEZ-MOI mais pas livré au moment de cette prise) — `FileNotFoundError`, rc=2.
* durée : 449 s (13:33:47–13:41:16), prévu ≤ 900 s (timeout), tenue

nvidia-smi avant/après identique (seul PID 4286, service permanent).

## Suite
1. Rejouer dans un trou plus calme (charge hôte stable) pour trancher la dérive ; ou 3e+4e paires si un poste stable ne se dégage pas en 2.
2. `equiv-sampler.py` à livrer par Laurine avant de pouvoir juger l'équivalence des ids.
