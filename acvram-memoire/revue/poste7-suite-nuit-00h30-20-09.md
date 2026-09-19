# poste7 — il est 00 h 27 à l'horloge machine, pas 07 h 25 : la nuit a encore 7 h 30 ; ordre des 7 h 30 restantes — niveau 2 (1 min qui nomme le tenseur figé), C5-b (fenêtre), C15 niveau 3 Coder (le pas b=12 contre vLLM), C13-c réécrit demain, FlashInfer/FlashMLA au bilan (20/09, 00 h 30, horloge machine lue)

Source : `date` = 20/09 00 h 27 ; poste1 00 h 25 (« bilan de fin de nuit 07 h 25 machine » — l'heure est celle de mes titres d'avant l'erratum, pas de la machine) ; `chantier-c13c-19-09` (5834c1b2 : plomberie forme 2, bras bf16 64 × 64 impossible sous 99 Ko : 147 Ko même en bf16, la structure qui tient est une réécriture ≥ 2 h) ; `poste1-c5b-k-par-canal` a77eb722 (rendu à sec, 9/9, chaîne carte) ; test carte `test_mla_prep_batch_sous_rejeu_de_graphe_suit_lens` (poste1-c15) ; fiches niveau 3 GLM (≤ 700 nœuds) et Coder (410-440 nœuds, 1 550-1 650 t/s prédits) ; `verdict-c9-m0-19-09`.

## 1. Heure — erratum bis
Mes addenda « 01 h 00 » à « 01 h 35 » de cette nuit valent ≈ 00 h 15 à 00 h 26 ; à partir de cette note chaque heure écrite est lue sur `date`. Le bilan poste7 reste à **07 h 30 machine**, dernier push chef **07 h 45 machine** ; personne ne clôt avant.

## 2. C13-c — fermé pour la nuit par la clause, réécrit demain, question à l'utilisateur au bilan
Le bras de structure ne peut pas se lancer (q + tuile de clés entières = 147 Ko en bf16 > 99 Ko) et le plafond à 32 × 64 est ≤ 1,8 × (> 10 ms/appel par arithmétique) → **clause appliquée : C13-c fermé cette nuit**. La forme qui tient est nommée par poste1 — **q par tranches BK en registres, un accumulateur [BM, rank]** — c'est une réécriture (≥ 2 h), chantier du jour suivant avec le scellé de la forme 2 (cœur ≤ 70 ms, prefill ≥ 9 500, juge contre l'einsum TF32) et la structure prouvée d'abord par une sonde de temps à bras bf16 (≤ 2 ms/appel) **avant** toute exactitude. La question d'un noyau MLA de prefill externe (FlashInfer / FlashMLA, dépendance : engage) va à l'utilisateur **dans le bilan**, avec les deux chiffres (7 268 aujourd'hui, ≈ 10 000 promis par la réécriture, 18 117 vLLM).

## 3. Ordre des 7 h 30 restantes
| rang | quoi | qui | durée | ce qui décide |
|---|---|---|---|---|
| 1 | **niveau 2** : test carte qui nomme le tenseur figé de `mla_prep_batch` (capture len=p, rejeu p+1, p+2, 100, 300 contre le jumeau) → correctif en place → rejeu cinq juges | poste2 1 min → poste1 → poste2 10 min | ≤ 1 h | `=2` au défaut : GLM b=1 −1,0 ms de noyaux, −600 nœuds ; préalable du niveau 3 GLM |
| 2 | **C5-b** fenêtre (chaîne d'poste1 : au bit vs jumeau, `ppl-decode-kv` 3 tranches jeton / canal / bf16, `certifie` ABAB, capture) | poste2 | 20 min | scellé de `poste7-c5-kv-int8-faux` § 3 : ΔPPL(canal) ≤ +0,002 contre bf16, pas b=12 ≥ défaut − 1 %, capture 5/5 ; +3,1 % du KV accepté (pas 1,5 : dit) → défaut |
| 3 | **C15 niveau 3 Coder** (fiche : 410-440 nœuds, 1 550-1 650 t/s) — code maintenant, sous-agent | poste1 | 3-4 h | scellé : nœuds ≤ 450, Coder b=12 servi sous 2 700 **≥ 1 550 · J net ≤ 0,175**, jetons au bit si seul l'ordre des lancements change / logits ± 1 ulp bf16 sinon, `ppl-decode-kv` avec SE, capture 5/5 ; fenêtre poste2 20 min si le commit arrive avant 06 h 30 — c'est la cellule qui rattrape vLLM (1 626 · 0,136) |
| 4 | cellule llama.cpp 119B b=1 experts en RAM (C9) ; M1 trace Coder dans un trou | poste2 | 20 min + 1 h | dit si le cache d'experts vaut un chantier (h) ; C9-charge = code, reprise |
| 5 | niveau 3 GLM après niveau 2 ; C15-prefill fiche ; C13-c réécriture | poste1 | demain | — |

Pas de nouvelle fenêtre pour : C13-c, C17, C1, NARROW, niveau 2 TF32, Mesure 2 — tous fermés cette nuit avec leurs chiffres.

## Ordre
* **poste2** — rang 1 (1 min, maintenant), 2, 4, puis rang 3 à son commit ; verdicts aux six lignes.
* **poste1** — rang 1 (correctif dès le tenseur nommé), rang 3 en sous-agent maintenant ; C13-c en fiche seulement (structure nommée, sonde de temps d'abord) ; aucune « fin de nuit » avant 07 h 30 machine.
* **chef** — ETAT : heure machine 00 h 30, nuit continue, ordre § 3 ; C13-c fermé cette nuit (structure), question FlashInfer au bilan ; fusions à leurs verdicts ; INDEX ; commit + push.
