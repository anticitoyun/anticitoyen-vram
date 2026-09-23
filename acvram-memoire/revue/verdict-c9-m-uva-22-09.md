# C9 M-UVA — TENU, 23,6 Go/s — 22/09 (Manon)

* instrument : `outils/gpu/mesure/c9-m-uva.py --rep 200` (harnais dédié Océane, remplace le chronométrage non concluant de `verdict-m-uva-22-09`), sous carte.sh
* commit : main à jour, worktree manon-w-21-09
* régime : 4 experts réels 119B w1 [2048,4096] nvfp4, 18,9 Mo/pas, 200 pas après 20 de chauffe, `nvfp4_gemv_grouped_table` G=4
* scellé (Océane, § 3.0) : prédit 15,0-19,0 Go/s ; vaut la peine ssi ≥ 17,0 ; arrêt si < 13,0 ; contrôle d'exactitude UVA==VRAM au bit
* mesuré : **exactitude TENUE** (sorties UVA==VRAM au bit) ; **UVA 799,1 µs (p90 803,4) → 23,6 Go/s (p90 23,5)** ; témoin VRAM 15,5 µs → 1216 Go/s (rapport UVA/VRAM 0,0194)
* verdict : **TENU — 23,6 Go/s ≥ 17,0**, au-dessus même de la fourchette prédite (15-19). Condition zéro-copie du § 2 remplie côté débit ; reste à mesurer h_pin(119B) réel (S1 ≥ 30 j/s si h_pin ≥ 0,72) pour trancher C9 dans son ensemble.
* durée : 1,5 s de calcul, < 1 min de carte

## Suite
La condition « vaut ≥17 » est remplie. C9-S1 dépend maintenant de h_pin mesuré sur le vrai 119B (pas le proxy 30B), hors de cette pièce.
