# 277e — ngram corrigé sous le nombre de pas de none : VERDICT (poste5 27/09 09 h)

* instrument : `scratchpad/poste5-p277e-27-09/pas277e.py` (5 × 32 gloutons, graphes, 2e passe, garde remise à zéro par invite) ; tests carte 277 et 277e
* commit : poste5-277e 0df186053 (code) ; témoin worktree détaché 4a32d7320 (code 277fix = main avant 0.7.6)
* régime : b = 1, Coder-30B-A3B-nvfp4-qkvo-i8c et Qwen3.8-27B-mixte-i8c, ngram k = 4, `ACVRAM_SPEC_REPOS` 2 (défaut) et 0
* scellé : `revue/poste5-piece277e-scelle-27-09.md` (+ avenant prise 2, avant mesure)
* mesuré : pas ngram < none sur les deux alias (Coder 138 / 160, mixte 128 / 160) ; temps Coder 1,049 × none, mixte 0,977
* verdict : **pas tenu ; débit Coder NON amélioré → issue (b)** — le pas n'était pas le bon critère ; pas de retour au défaut
* durée : prévu ≤ 40 + 15 min ; tenu prise 1 08:17-08:23 (6 min 13 s), prise 2 tenue=278 s (journal carte.sh)

## Mesures (prise 2 ; ms = moyenne des invites 1-4, 2e passe)
| alias / côté | pas (none 160) | max / invite | temps ngram / none | proposés / acceptés | sorties |
|---|---|---|---|---|---|
| Coder témoin 277fix | 227 | 55 | 1,023 (129,6 / 126,7) | 106 / 28 | ≠ none invites 0, 3 |
| **Coder 277e REPOS=2** | **138** | 31 | **1,049** (133,0 / 126,8) | 57 / 9 | ≠ none invite 0 |
| Coder 277e REPOS=0 | 113 | 27 | 1,031 (132,1 / 128,0) | 95 / 24 | ≠ none invites 0, 3 |
| mixte témoin 277fix | 227 | 56 | 0,985 | 84 / 32 | = none au bit |
| **mixte 277e REPOS=2** | **128** | 31 | **0,977** (676,5 / 692,4) | 44 / 20 | = none au bit |
| mixte 277e REPOS=0 | 105 | 25 | 0,986 | 88 / 33 | = none au bit |
Tests carte (prise 1, `p1/`) : 277 deux volets **3/3 verts** sur 277e (inchangé) ; 277e **vert** sur 277e, **ROUGE** sur
main 1b59ed0bd (Coder 41/39/52/24/55, mixte rouge aussi) : il casse si l'on réintroduit l'alternance.
Prédictions : Coder REPOS=2 125-155 pas → 138 tenu ; REPOS=0 120-150 → 113, FAUX (mieux) ; mixte < 160 tenu, temps
0,90-1,30 tenu ; **Coder temps 0,85-1,03 × none → 1,049, FAUX** (REPOS=0 1,031, à la limite).

## Lecture
* Chaque pas livre ≥ 1 jeton (max 31 / invite) : l'alternance amorce/vidage de la 277fix est éteinte.
* Le temps ne suit pas le pas. Sur le Coder, une vérification q_len 5 coûte plus qu'un pas simple (MoE : plus d'experts
  lus) et l'acceptation reste basse (9 / 57 à REPOS=2, 24 / 95 à REPOS=0) : le ngram ne paie pas sur cet alias, quel que
  soit le regroupement en pas. La Q26 (correction optimiste sans vidage) ne rendrait que le recouvrement perdu, pas
  cette acceptation : gain attendu ≤ 1-2 %, **non recommandée** sur cette base. Sur le mixte, gain de 1,4-2,3 %.
* Échantillon unique par côté (none stable à ± 0,5 ms ; l'écart REPOS 2 contre 0 sur le Coder, 1,049 contre 1,031, est
  dans ce que je ne sais pas départager sans répétition) : pas de changement du défaut REPOS=2 scellé.
* Q30 (poste4) : `speculative_disable_by_batch_size` de vLLM coupe sur la charge, pas sur l'acceptation ; notre
  `GardeSpeculation` coupe sur le rendement en jetons/pas — elle a coupé le ngram dès la 1re passe en prise 1 (défaut
  d'instrument de ma prise 1, pas du moteur). Elle compte des jetons, pas du temps : sur le Coder elle laisse passer un
  ngram qui fait moins de pas mais coûte plus de temps.

## Défauts d'instrument (assumés)
Prise 1 : garde non remise à zéro entre passes (débit non mesuré). Prise 2 1re tentative : refus de ma garde, main passé en
0.7.6 → témoin sur worktree détaché. Noms de fichiers : `${alias:0:5}` identique pour les deux alias, les JSON Coder ont
été écrasés par le mixte (renommés `pas-mixte-*`) ; les chiffres Coder viennent de `prise.log` (ligne imprimée par
`pas277e.py`), complets.

## Proposé à chef
Fusion possible (hors défaut ngram, test 277e cassant, test 277 inchangé vert). Le retour du ngram au défaut n'est pas
justifié par le débit sur le Coder (+3 à +5 % de temps) ; sur le mixte, +1,4-2,3 % — à juger avec la qualification
mmlu_v275 de poste2. Si l'on veut un critère de garde en temps plutôt qu'en jetons/pas : pièce à part.
