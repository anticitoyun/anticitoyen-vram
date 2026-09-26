# Reconversion Coder --alpha-commun-experts — REFUS (garde ≥512 observations/expert) — 22/09 (poste2)

* instrument : `outils/carte.sh .venv/bin/python -m acvram convert .../Qwen3-Coder-30B-A3B-Instruct-bf16-hub -o .../Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha-22-09 --alpha-commun-experts`
* mesuré : calibration terminée (48/48 couches), statistiques relevées pour 16 299 tenseurs — **`observations par expert : min 1, p10 3, médiane 144, max 16310 ; 3522/5369 sous 512 (65,6 %), jamais routés 0`** — **REFUS explicite du convertisseur** : « 3522 expert(s) sous 512 observations (minimum 1, jamais routés 0) — corpus insuffisant, il faudrait ×512,0 de jetons de calibration ».
* verdict : **REFUS, aucun alias produit** — le convertisseur applique désormais le seuil de stabilité ≥512 observations/expert établi ce matin (`verdict-piece25a-stabilite-awq-post-fix-22-09.md`) et le corpus de calibration par défaut (32×512=16 384 jetons) est trop court pour l'atteindre sur 65,6 % des experts. Ce n'est pas un bug — c'est le garde-fou attendu, appliqué pour la première fois à une conversion réelle aujourd'hui.
* durée : ~9 min de carte (calibration complète avant le refus, pas de gaspillage de conversion)

## Suite
Deux options, à trancher par le groupe (pas de décision unilatérale) : (a) relancer avec `--obs-min 0` (accepte la conversion malgré le corpus insuffisant, en le nommant dans la fiche — dégrade la garantie de stabilité AWQ sur 65,6 % des experts) ; (b) relancer avec `--calib-seqs` plus grand ou un corpus différent pour atteindre 512 observations sur plus d'experts (×512 jetons de calibration nécessaires selon le message — coûteux). Chaîne ABBA/familles-noyaux/KL/PPL suspendue tant que l'alias n'existe pas.
