# Pièce 25(a) awq-stabilite-experts, rejoué après correctif (bd5c7932) — TENU, seuil ≥512, pas 32-127 — 22/09 (poste2)

* instrument : `outils/awq-stabilite-experts.py /mnt/AI_GENERATOR/models_acvram/Qwen3-VL-30B-A3B-awq-dequant-bf16 --jetons 100,1000,10000 --json`, sous carte.sh
* commit : main/poste2 à jour (8b68082b), correctif lecteur à 3 dispositions (bd5c7932) déjà fusionné
* régime : 3 collectes (100/1000/10000 jetons), tolérance 0,10, échantillonnage stratifié par classe de jetons routés
* scellé (poste1) : seuil de stabilité attendu classe 32-127 ; critère groupe (22/09, avis croisés) : stable à partir de ≥256 obs/expert (erreur-type ≈6 %), ≥512 pour les hétérogènes
* mesuré : 240/240 experts lus (17 235 tenseurs de référence, 0 non lu — correctif tenu). Distribution par classe (n, écart médian, p90 — min/p10/max non émis par cet instrument) :

| classe (jetons routés) | n échantillons | écart médian | écart p90 |
|---|---|---|---|
| 1-7 | 175 | 0,343 | 0,591 |
| 8-31 | 76 | 0,210 | 0,344 |
| 32-127 | 69 | 0,153 | 0,287 |
| 128-511 | 21 | 0,128 | 0,161 |
| ≥512 | 3 | 0,084 | 0,084 |

Référentiel : 17 235 tenseurs répartis 1-7:1962 · 8-31:2421 · 32-127:3366 · 128-511:3900 · ≥512:5586.

* verdict : **TENU** — l'instrument est réparé (0 → 240/240 experts), mais **le seuil mesuré (≥512, médiane 0,084 ≤ tolérance 0,10) est PLUS TARDIF que les deux prédictions** : au-dessus de la classe 32-127 d'poste1 (médiane 0,153, encore 53 % au-dessus de la tolérance) ET au-dessus du plancher ≥256 du groupe (aucune classe entre 128 et 511 n'atteint 0,10 : 128-511 médiane 0,128, encore 28 % au-dessus). Seule la classe ≥512 est sous tolérance, avec seulement n=3 échantillons — signal faible mais net (croissance monotone régulière de l'écart à mesure que n augmente sur les 5 classes, pas de rupture brutale qui suggérerait un artefact).
* durée : ~3 min de carte (75+? s de collecte × 3, verrou rendu propre)

## Suite
Retenir ≥512 comme seuil de stabilité AWQ pour ce modèle (30B-A3B), plus conservateur que le ≥256 proposé — n=3 dans cette classe est faible, un rejeu avec plus de jetons routés (au-delà de 10 000) donnerait un n plus solide si le groupe le juge utile. File : reconversion 30B-VL mediane_couche → (c)+P3 → KL acvram/bf16 → énergie 4 moteurs.
