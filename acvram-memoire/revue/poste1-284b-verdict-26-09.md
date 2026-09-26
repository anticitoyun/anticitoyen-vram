# 284 b — préfill « une par une » réordonné couche par couche : AU BIT, TTFT b=12 du mixte −54 % — PARTIEL (poste1, 26/09)

* instrument : `tests/test_prefill_tranches_284.py` (au bit contre `ACVRAM_PREFILL_TRANCHES=0`, logits du préfill + jetons, b=1
  et b=3 ; cassant tranches croisées ; 2e tour repris à l'instantané) ; `scratchpad/poste1-p284-27-09/prise-284b.sh` (tests sur
  carte puis TTFT servi, `client-carte-274.py`, A = main worktree jetable, B = 284 b, A B B A, port neuf par bras, passe nulle
  bloquante sur le compteur `prefill_tranches`)
* commit : poste1-284 1f957774b ; A = origin/main 0286a4207
* régime : Qwen3.8-27B-unsloth-mixte-i8c, cache de préfixe ON (défaut), -lgc 2700, ACVRAM_ECO=off,
  `serve --max-batch 16 --max-model-len 4096 --speculative none`, invites réelles distinctes ~470 jetons
* scellé : `scratchpad/poste1-p284-27-09/scelle.md` addendum 3 (dc67c9a63) + addendum 4 (1f957774b, après une mesure nulle)
* mesuré : tests 41 passés sur carte ; TTFT b=1/b=12, deux passages par bras
* verdict : **PARTIEL** au seuil scellé — B b=12 = 2,55 s dans les deux paires (cible ≤ 2,4 ; FAUX au-delà de 2,9) ; au bit tenu
* durée : prévu ≤ 30 min ; tenu 439 s (journal `tenue=`) ; 1re prise (nulle) 490 s

## Chiffres (TTFT p50 / p95 ms ; débit servi t/s)
| bras | b=1 | b=12 | débit b=12 | `prefill_tranches` |
|---|---|---|---|---|
| A main | 480 / 483 ; 464 / 501 | **5 513 / 5 559 ; 5 495 / 5 526** | 110 ; 123 | — |
| B 284 b | 481 / 521 ; 465 / 489 | **2 558 / 2 602 ; 2 548 / 2 751** | 202 ; 215 | 6 ; 6 |
b=12 : −53,6 % ; b=1 : inchangé (une séquence ne prend pas le chemin : `len > 1`). Repère 284 : sans cache de préfixe, 1,92-2,04 s.

## Ce qui a changé (au bit)
* `Engine._prefill_tranches` (runner.py) : la boucle « une par une » en deux VAGUES — toutes les séquences jusqu'à leur
  frontière, puis les instantanés (séquence par séquence, comme avant), puis toutes jusqu'au bout — chaque tranche construite par
  le même `_build_batch` qu'avant. `ACVRAMModel.forward_tranches` (model.py) : couche par couche, chaque tranche passe SEULE
  (mêmes M, mêmes appels), dans `depaquetage_partage(seuil_partage=False)` : la déquantification d'une couche est faite une fois
  par vague au lieu d'une fois par tranche, et le seuil GEMV int8 ne bascule PAS à 16 (243, hors bit). `_sortie` : la fin de
  `forward`, factorisée et partagée.
* Refus (boucle d'avant, inchangée) : image, deepstack, résidu différé, synchronisation de diagnostic, proposeur MTP actif (il
  relit l'état caché dans l'ordre des appels).
* `ACVRAM_PREFILL_TRANCHES` (défaut 1, 0 = témoin) dans `regime.VARIABLES` ; compteur `prefill_tranches` dans /metrics.

## Écart à la cible (2,55 contre 2,4 s)
Deux vagues = deux déquantifications par couche et par pas, plus les instantanés : ≈ 0,55 s au-dessus du « sans cache » (C).
Reste au bit : une seule vague pour les séquences dont la frontière est commune (même pas, même longueur arrondie) — non fait.

## Leçons
* La 1re mesure était NULLE : mon garde refusait toute tête MTP chargée, et l'alias mixte en charge une même sans spéculation ;
  les tests (modèle sans MTP) ne pouvaient pas le voir. Le compteur de prise, rendu bloquant, l'a vu ; la mesure a été refaite.
* Contrôle du 2e tour (condition 3) : `test_second_tour_reprend_a_l_instantane` — un 2e tour qui partage l'amorce a
  `cached_len` ≥ la frontière.
