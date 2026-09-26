# Verdict — 142 réserve (1) : capacité KV de A contre B, trois modèles — 24/09 (poste1)

* **instrument** : `scratchpad/poste1-p142-24-09/capacite/` — `plan-kv.py` (à sec, plan du manifeste), `charge-kv.py`
  (load_model réel 8 × 2 560, capacité = `_kv_blocks_per_device(plan borné)` × 16, un bras par processus), `prise-kv-fige.sh`
  (set -euo pipefail, copie figée) ; B = `ACVRAM_PROJ_MARLIN=1 DOUBLES=` et `ACVRAM_PROJ_MARLIN_CAPACITE=16` (garde neutralisée
  pour LIRE la capacité, pas pour servir)
* **commit** : 38b6acdb (poste1-mtp) ; carte par carte.sh (ACVRAM_NOM=poste1-p142-capacite-kv, obtenue après 1 792 s d'attente
  derrière poste5-139-c-b1) ; compute-apps début = fin (4627, 5 606 Mio, la 3080 Ti)
* **scellé** : `capacite/scelle-kv.md` (avant la prise) — prédit : écart B − A = `_octets_marlin` (réserve de la tête) quand la
  borne VRAM agit ; FAUX si A ≥ 20 480 sur gemma31/qwen32, ou si B − A dépasse `_octets_marlin` de > 10 %
* **mesuré** (8 × 2 560 = 20 480 jetons demandés) :

| modèle | A jetons | B jetons | budget KV (A = B) | o/jeton | libre après A / B | alloué A / B |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | **20 480** | **20 480** | 0,630 Gio (= la demande) | 33 024 | 13,43 / 11,14 Gio | 16,97 / **19,11** Gio |
| gemma4 31B | **3 824** | **3 824** | 1,768 Gio | 495 360 | 3,57 / 7,46 Gio | 21,18 / **22,51** Gio |
| qwen32 | **14 256** | **14 256** | 1,768 Gio | 133 120 | 9,65 / 9,73 Gio | 20,16 / 20,18 Gio |

* **verdict** :
  * **B ne réduit PAS la capacité KV** : A = B au bloc près sur les trois modèles. La réserve (1) de la 142 était fausse. Le
    refus de B à 8 × 2 560 venait d'une **asymétrie de garde** : `_verifier_memoire_marlin` (loader.py:1239, 1252) compare la
    capacité à la demande sous PROJ_MARLIN seulement ; A a la même capacité (14 256 et 3 824 < 20 480), mais rien ne la
    vérifie, et A sert en silence avec moins que demandé.
  * **Ma prédiction de cause était fausse** : `_octets_marlin` (loader.py:1755, qui réserve la tête : 0,666 / 0,061 /
    0,408 Gio) grossit bien la réserve de B, mais la borne VRAM (`_borner_kv_par_la_vram`, loader.py:1125) n'agit pas ici
    (aucune ligne « borné »). La réserve n'est donc pas ce qui limite le KV.
  * **Cause réelle, commune à A et à B : le planificateur**. `auto_plan` (tiering.py:887-906) essaie les fractions
    `[0,06 … 0,55]` de la VRAM pour le KV. Une fraction est jugée faisable si `kv_max_tokens >= max_model_len`
    (tiering.py:896), c'est-à-dire si **UNE** séquence tient, pas les `max_concurrent_seqs`. Le classement `_rang`
    (tiering.py:966-973) ne regarde que l'exil puis le débit estimé, sans la taille du KV. Les débits sont égaux, et `max`
    garde le premier à égalité : c'est la fraction 0,06. Contrôle arithmétique : 0,06 × 29,47 Gio = **1,768 Gio**, le budget
    réel de gemma31 ET de qwen32 (deux modèles différents, le même budget) ; au manifeste, 0,06 × 30,13 = 1,808 (qwen32) et
    0,15 × 30,12 = 4,517 (gemma31). Qwen3.8 n'est pas touché, car sa demande (0,63 Gio) tient sous 0,06.
  * **Aucune copie évitable ne limite le KV aujourd'hui**. En revanche, B **alloue plus** que A : +2,14 Gio sur Qwen3.8,
    +1,33 Gio sur gemma31, +0,02 sur qwen32, alors qu'il n'y a pas de doubles. Cela ne coûte rien tant que le KV est plafonné
    par le planificateur, et deviendra une perte de capacité dès que le KV sera borné par la VRAM (après le correctif
    ci-dessous). **Non attribué** ; hypothèses : stockage naturel gardé vivant par les vues de pile (GDN, qkv) sur Qwen3.8, tête
    liée nvfp4 de gemma. À mesurer par un inventaire des NVFP4Tensor après chargement (qweight naturel ET `_marlin_dense`).
* **correctif prédit (non codé, au chef)** : dans `auto_plan`, exiger `kv_max_tokens >= max_model_len × max_concurrent_seqs`
  pour `ctx_ok` (tiering.py:896), le cas « aucune fraction ne tient » retombant déjà sur `_plan_kv_maximal`. Autre solution :
  départager `_rang` par `min(kv_tokens, demande)`. Effet prédit à 8 × 2 560 : qwen32 A = B = **20 480** jetons (2,54 Gio ;
  9,65 Gio libres) ; gemma31 monterait jusqu'à la borne VRAM (entre 3 824 et 9 776 ; la marge de 1,6 Gio plus la réserve de
  2,03 Gio en décident, à mesurer) ; Qwen3.8 inchangé. Risque : le changement touche tous les modèles, donc aussi A ; test
  d'équivalence du plan (Qwen3.8 inchangé) et ligne `kv_budget` du régime à relever.
* **durée** : prise 6 chargements, < 6 min après obtention de la carte
