# Pièce 156 (a) — scellé (poste5, 24/09, écrit AVANT la prise) : une seule marge KV

Ordre de l'utilisateur relayé par chef (régler les dettes connues). Bead `anticitoyen-vram-ba9` (poste1, 146e).

## À sec

* Deux marges pour le même risque (carte trop pleine au préfill ou à la capture) :
  `_borner_kv_par_la_vram` max(1,5 Gio, 5 %) + réserve (`loader.py`, `_KV_MARGE_MIN/_PART`) ; `_reajuster_plan`
  max(2 Gio, 7 %) + réserve, en dur (`loader.py`, « marge pour le contexte CUDA… »).
* Les 7 % datent du 08/09 (2866501b) : un 70B chargé à 99 % tombait en OOM au premier préfill. Ce cas a depuis sa
  propre réserve, nommée (`reserve` = `activations_prefill_bytes`, 1aa767bc, 17/09). Les deux points de plus ne servent
  plus qu'à retirer du KV : c'est ce site qui borne gemma31 à 13 408 (146e).
* Changement : `_marge_carte(capacite, reserve)` = max(`_KV_MARGE_MIN`, `_KV_MARGE_PART` × capacité) + réserve, lue par
  les deux sites. Valeur retenue : **1,5 Gio / 5 %** (la capture échoue sous ~1 Gio libre). La base reste différente :
  `_reajuster_plan` part d'une capacité déjà nette de build_tiers (800 Mio + 3 %), la borne part de `libre` brut. Je ne
  la change pas ici : le commentaire du site interdit `capacite = libre`, et un changement de base serait un autre geste.
* Sortie du modèle : inchangée (seuls les budgets bougent). Un KV plus grand change la capacité, pas l'arithmétique.

## Instrument

1. Tests processeur : `tests/test_marge_kv_unique.py` (neuf), `test_budget_exil_prefill.py` (montage recalé au bord
   de la marge unique), `test_kv_plancher_exil.py`, `test_garde_kv_146.py`, `test_planificateur_kv_146.py`.
   Bras cassants dans une copie : (i) `max(2 * 2**30, int(0.07 * capacite))` remis en dur dans `_reajuster_plan` →
   ROUGE `test_l_exil_suit_la_marge_unique` ; (ii) la borne en dur à 0,05 → ROUGE `test_la_borne_du_kv_suit…`.
2. Capacité KV : `chauffe.py` de la 146 (défaut servi, 8 × 2 560 jetons, 64 générés chacune), avant (main fusionné,
   commit parent) et après, sur gemma31 (`gemma-4-31B-it-nvfp4-vision`), qwen32 (`DeepSeek-R1-Distill-Qwen-32B-…-nvfp4`)
   et Qwen3.8 (`Qwen3.8-27B-nvfp4`). Relevés : capacité, budget, pic alloué, nvidia-smi, finies/tronquées, régime.

## Prédiction

* **gemma31 : 13 408 → 14 300-14 800 jetons** (marge 7 % → 5 % sur ≈ 29,6 Gio nets : +0,57 Gio ; 495 360 o/jeton ⇒
  +≈ 1 240). Si la borne du KV (base brute) mord avant, le gain sera plus petit, et je dirai laquelle.
* **Qwen3.8 et qwen32 : inchangés à 20 480** (déjà à la demande, 8 × 2 560 ; le KV ne monte pas au-dessus de la cible).
* Les trois : 0 exil, 8/8 finies, 0 tronquée, pas d'OOM, régime NOMINAL.

## Issues nommées

* (a) gemma31 inchangé → ce n'est plus `_reajuster_plan` qui borne ; relire les lignes « borné par la VRAM libre ».
* (b) OOM, exil ou DÉGRADÉ sur un des trois → 1,5 Gio / 5 % trop mince pour le site d'exil ; alors la constante unique
  passe à la valeur qui tient (7 %), je chiffre ce que Qwen3.8 et qwen32 perdent, et c'est chef qui tranche.
* (c) Le 70B (Llama-3.3-70B-nvfp4, cause des 7 %) n'est PAS rejoué ici : la réserve de préfill le couvre sur le papier
  (`test_budget_exil_prefill`), pas sur la carte. Risque résiduel nommé.
