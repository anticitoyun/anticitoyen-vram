# 277e — ngram corrigé sous le nombre de pas de none : SCELLÉ (poste5 27/09, avant mesure)

Ordre chef 27/09. Branche poste5-277e depuis origin/main 1b59ed0bd (277fix). Pas de changement de défaut.

## Cause (à sec, fichier et ligne)
En spéculatif, un proposeur muet fait retomber `_speculative_decode` sur `_plain_decode` → `_pipeline_amorcer`
(pipeline.py, pas lancé, **0 jeton livré**) ; au `step()` suivant, un pas est en vol → la 277fix le vide **seul**
(runner.py:1652 sur main). Deux pas de `step()` par jeton en veille, d'où 24-55 pas / 32 (départage 277fix : 211 contre 160).

## Correctif
`_pas_speculatif` (pipeline.py) : (1) vidage et spéculation dans le même pas ; (2) une amorce qui ne livre rien est
suivie de la suite du pipeline dans le même pas (chaque pas livre ≥ 1 jeton) ; (3) hystérésis
`ACVRAM_SPEC_REPOS` (défaut 2) : après un repli, 2 pas en recouvrement avant de revider pour reproposer.
Q26 (correction optimiste de l'état hôte sans vidage) N'EST PAS écrite : ouverte seulement si l'issue (b) ci-dessous.

## Instrument
`scratchpad/poste5-p277e-27-09/pas277e.py` : 5 invites × 32 gloutons, graphes, 2 passes par mode sur le même moteur,
2e retenue ; ms par invite, débit sur les invites 1-4. Côtés : main 4a32d7320 (code 277fix, témoin), 277e REPOS=2, 277e
REPOS=0 ; Coder puis mixte-i8c. Tests carte : 277 (deux volets, inchangé) et 277e (pas ≤ none par invite, total < none).

## Prédictions
| | pas / 5 × 32 (none 160) | débit ngram / none (ms 1-4) |
|---|---|---|
| Coder témoin 277fix | 190-230 (vu 211) | 0,95-1,10 × le temps |
| Coder 277e REPOS=2 | **125-155**, chaque invite ≤ 32 | temps 0,85-1,03 × none |
| Coder 277e REPOS=0 | 120-150 (≤ REPOS=2) | temps 0,90-1,05 × none |
| mixte 277e REPOS=2 | **< 160**, chaque invite ≤ 32 | temps 0,90-1,30 × none (vérif. q_len 5 = 35,8 ms contre ~15,7) |
Sorties : mixte = none au bit sur les trois côtés ; Coder : divergences top1/top2 marge ≤ 0,5 (test 277).
Test 277e sur main : ROUGE sur le Coder (211 > 160) — condition d'entrée.

## Issues nommées
(a) pas < none sur les deux alias, sorties tenues, test 277e rouge sur main : **tenu**.
(b) pas < none mais temps ngram > 1,03 × none sur le Coder : le pas n'était pas le bon critère ; le recouvrement perdu
coûte → Q26 (correction optimiste) justifiée, pas de retour au défaut.
(c) pas ≥ none sur un alias : correctif faux ou incomplet (repli non couvert) — pièce non tenue.
(d) mixte ≠ none au bit : BOGUE, arrêt, rien en main.
(e) test 277e VERT sur main : le test ne contrôle rien, à réécrire avant tout verdict.
Budget : prise ≤ 40 min de carte ; pas de seconde prise sans ordre.

## Avenant 27/09 08 h 3x — prise 1 rendue, prise 2 avant mesure (défaut d'instrument)
Prise 1 (0df186053, `scratchpad/poste5-p277e-27-09/p1/`) : tests carte rendus — 277 3/3 vert sur 277e ; 277e vert sur
277e, ROUGE sur main (Coder 41/39/52/24/55, mixte rouge aussi) : condition d'entrée tenue, issue (e) écartée. Débit NON
mesuré : en 2e passe la garde de rendement (`GardeSpeculation`) avait coupé la spéculation (Coder : 0 proposé sur les
trois côtés). Prise 2 : seul changement, garde remise à zéro avant chaque invite ; prédictions du tableau inchangées.
Écart à « pas de seconde prise sans ordre » : défaut de mon instrument, prise ≤ 15 min, assumé ici.
