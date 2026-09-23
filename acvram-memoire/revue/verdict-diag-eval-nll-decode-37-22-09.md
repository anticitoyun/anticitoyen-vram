# diag-eval-nll, bras decode (pièce 37) sur Gemma 4 31B — INVALIDE, auto-détecté — 22/09 (Manon)

* instrument : `outils/carte.sh .venv/bin/python outils/gpu/mesure/diag-eval-nll.py gemma-4-31B-it-nvfp4-vision --jetons 300`, sha `bf8afbdc` (bras decode ajouté par Océane)
* commit : main à jour
* mesuré : `PPL decode 185080,69 · NLL médiane 11,7749 nats · 1re divergence eval/decode position 20 (|Δ| max 7,9595)` ; **contrôle de montage propre à l'instrument (position 1, contexte d'un seul jeton) : FAUX** — `Δ = 0,0142 > 1e-2` entre decode et eval, alors que les deux devraient coïncider exactement à cette position (aucun cache, même calcul).
* verdict : **INVALIDE — l'instrument se rejette lui-même** (sortie littérale : « Le bras est mal monté, il ne juge rien »). La NLL médiane (11,77 nats) est numériquement proche de la prédiction P4 (≈12 nats), mais je ne la publie PAS comme résultat exploitable — le contrôle de montage échoue, donc rien ne garantit que le reste de la trace decode mesure ce qu'il prétend mesurer.
* durée : ~2 min de carte

## Suite
Bras decode à corriger par Océane (le montage position 1 doit coïncider au bit près avec eval, contexte identique, aucun cache en jeu — l'écart de 0,0142 suggère un détail d'initialisation différent entre les deux chemins). P4/P5 restent non tranchés.
