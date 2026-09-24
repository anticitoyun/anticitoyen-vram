# Verdict — pièce 139 : unsloth/Qwen3.8-27B-NVFP4 « mixed-precision » servi par acvram, face à NInfer (poste5, 24/09)

* **instrument** : `acvram.cli eval` (PPL de la 102, `par_fenetre`) + bootstrap apparié (`analyse-ppl.py`, fonctions de
  `poste4-piece138-sem-ppl-24-09/bootstrap_ppl.py`) ; banc chat de la 102 (`banc-chat-openai.py`), prise `prise-c.sh`
  ABBA×5 par b, serveur neuf par passe ; NVML `energie.py`, carte 0 (5090) seule
* **commit** : a294ec79 (branche poste5) — (a)/(b) sur bee69c40, même code moteur ; alias `Qwen3.8-27B-unsloth-mixte-i8c`
  converti à sec (`--passage-direct --no-awq --sans-vision`, `ACVRAM_HFQUANT_PAR_GROUPE=1`)
* **régime** : défaut, `prefill_int8=cublas+bf16(origine fp8 ×233)` relevé à chaque passe ; `-lgc 2700` posé pour les
  deux moteurs, plafond 400 W ; cpu-safe 100 début/fin ; llama-server de l'utilisateur sur la 3080 Ti (jamais touché)
* **scellé** : `revue/poste5-piece139-scelle-carte-24-09.md` + addenda 1 (faux, corrigé), 2 et 3, tous écrits avant leur prise
* **mesuré** : (a) jeton godet 1 « Paris. », VRAM serveur 26,8 Gio (b=8, 4096), poids 22,50 Go ; (b) **PPL 3,9609** ;
  (c) b=1 acvram **58,8 t/s** / 5,315 J, NInfer 74,6 / 4,350 ; b=8 acvram **245,6** / 1,245 J, NInfer 463,3 / 0,699
* **verdict** : (a) TENU (poids 0,4 Go au-dessus de la fourchette, tête MTP bf16 non comptée) ; (b) prédiction 4,05-4,25
  **NON TENUE par le bas** — apparié à NInfer **−5,67 %, z −3,63**, IC95 [0,912 ; 0,970] : écart conclusif ; (c) b=1
  **FAUX** (< 65), b=8 hors fourchette (< 270), non réfuté (≥ 240). **Formats différents** : acvram W4A16 (MLP 0-55)
  + **W8A16** (40 % des paramètres, int8 ré-encodé du fp8) contre NInfer W4A4 + **W8A8** — ce n'est pas le même format.
* **durée** : prévu ~45 min de carte / tenu : (a)+(b) 60 s, (c) b=1 8 min 07, b=8 8 min 22 (`carte.sh`) ; plus 6 prises
  d'échec et de diagnostic (OOM), toutes ≤ 2 min

## Table (c)

| b | moteur | t/s (σ, passes) | J/jeton net (σ) | horloge moy. | W |
|---|---|---|---|---|---|
| 1 | acvram mixte | 58,8 (0,1 ; 5) | 5,315 (0,008) | 2 637 | 386 |
| 1 | NInfer | 74,6 (0,1 ; 5) | 4,350 (0,018) | 2 368 | 399 |
| 8 | acvram mixte | 245,6 (0,6 ; 5) | 1,245 (0,002) | 2 554 | 381 |
| 8 | NInfer | 463,3 (0,3 ; 5) | 0,699 (0,001) | 2 385 | 400 |

**Régime du client** : le champ `regime` des lignes RESULTAT vaut « indisponible (ImportError …) » — le CLIENT
(banc) importait acvram depuis main et la garde d'arbre (a86fa1dd) l'a refusé ; il ne touche que ce champ
informatif. Le SERVEUR mesuré tournait sur la branche poste5 : `prefill_int8=cublas+bf16(origine fp8 ×233)` relevé
dans son journal à chacune des 10 passes acvram (`prise-c-b1.txt`, `prise-c-b8.txt`).

**Horloges** : écart 269 MHz (b=1) et 169 MHz (b=8) > 30 → au sens du scellé, cellule **non comparable à horloge
égale**. NInfer sature le plafond de 400 W et descend à ~2 370 MHz ; acvram tient une horloge plus haute. Corriger
l'horloge creuserait l'écart de débit au lieu de le réduire : acvram est plus lent même avec une horloge plus haute.

## Lecture

* **Qualité** : au même point de contrôle, acvram est plus proche du bf16 (3,6882) que NInfer (+7,4 % contre +14,1 %).
  C'est le prix des activations 4 et 8 bits de NInfer, pas un avantage de moteur : sur ces 40 % de paramètres, nos
  activations ne sont pas quantifiées.
* **Débit b=1** : 19,05 Go de poids lus par pas (nvfp4 8,42 + int8 10,63) à 58,8 t/s → 1,12 To/s effectifs. Ma
  prédiction supposait un int8 aussi efficace par octet que notre nvfp4 à 74 t/s. Mesuré : le GEMV int8 par canal
  tient ~1,1 To/s, le chiffre de la p57 que j'avais cité sans l'appliquer. Levier : le fp8 servi nativement (option C)
  ne réduit pas les octets, il faudrait un GEMV 8 bits plus rapide. Ou bien servir ces couches en nvfp4, mais ce ne
  seraient plus les mêmes poids.
* **b=8** : NInfer ×1,9 (tensor cores W4A4/W8A8) — même ordre que la 102 sur notre alias nvfp4 (×1,56).

## Défauts trouvés et corrigés en route (tous avant la mesure jugée, tests 390 verts, cassure 15/15)

Tour visuelle Qwen3_5 gardée puis refusée au chargement (`--sans-vision`, incohérence notée au dossier § 6) ; copie
int8 signée persistante du préfill cublas sur 233 tenseurs (+10,6 Go, OOM) → déquant bf16 marquée, marque propagée à
`stack_int8_linears` et `INT8Tensor.to`, repli bf16 ouvert au par canal par `vue_g128` au bit ; refus nommé modelopt
MIXED_PRECISION. Non fait : `tests/test_mm_mrope.py` (attend la carte, interblocage sous mon verrou) à rejouer hors verrou.
