# Pièce 237d — scellé À SEC (poste2, 26/09, ordre chef), mesure APRÈS 15 h (régime plein)

Suite de la 261b (`revue/poste2-piece261b-verdict-mcnemar-26-09.md`) : à n=50/100, McNemar ne
montre aucune différence P1/P0, mais l'IC de la récupération reste large (borne basse ≈ 93,8 %
sur GSM8K à n=50, testé sur les données de la 261b) — trop large pour conclure à ≥ 99 % ou même
≥ 97 %. Décision de chef : PAS de changement de défaut sans preuve POSITIVE ; n plus grand.

## Protocole (n plus grand, même appariement que la 261b)

* **GSM8K** : n = 250 (contre 50 en 261b).
* **MMLU** : 3 tâches × n = 150 chacune — `mmlu_flan_cot_fewshot_high_school_mathematics`,
  `mmlu_flan_cot_fewshot_professional_law`, `mmlu_flan_cot_fewshot_college_computer_science`
  (variante `fewshot`, split `test` — PAS `zeroshot`/`validation`, bogue de la 261b déjà
  corrigé, ne pas régresser).
* Graine 1234 fixe, mêmes questions P0/P1 (vérifié par égalité des `doc_id`, comme la 261b).
* Prises ≤ 30 min chacune (REGLES, régime plein après 15 h) — une prise = une tâche × un bras
  (P0 ou P1), reprise sur fichier déjà écrit (`prise-tache-261b.sh`, réutilisé tel quel,
  aucune modification nécessaire).
* Instrument : `mcnemar-ic-237d.py` (McNemar + IC bootstrap apparié de la récupération
  P1/P0, R=10 000 tirages, déjà vérifié sur les données de la 261b avant cette pièce — voir
  ci-dessous).

## Critère d'acceptation (chef, avant mesure)

**`PAR_LIGNE` passe à 1 par défaut en 0.7.1 SI, sur CHAQUE tâche (les 4) :**
1. McNemar p > 0,05 (pas de différence significative détectée) ; **ET**
2. la borne BASSE de l'IC à 95 % de la récupération P1/P0 est **≥ 97 % EN MOYENNE** sur les 4
   tâches (moyenne des 4 bornes basses, pas une seule tâche isolée).

**Sinon** : `PAR_LIGNE` reste à 0 par défaut, disponible en opt-in — pas de nouvelle mesure
automatique tant que chef ne la redemande pas.

## Prédiction (avant mesure)

* **McNemar** : je prédis p > 0,05 sur les 4 tâches (cohérent avec 261b à petit n, et avec la
  237b où KL(P0,HF) et KL(P1,HF) étaient quasi égales — rien n'indique un effet réel, seulement
  du bruit d'échantillonnage).
* **Borne basse IC95% ≥ 97 % en moyenne** : **incertain, prédiction proche de la limite** — à
  n=50 sur GSM8K seul (261b) la borne basse était à 93,8 %, sous 97 % ; à n=250 (5× plus), la
  largeur de l'IC (∝ 1/√n) devrait environ se diviser par √5 ≈ 2,24, ramenant une largeur
  d'ordre 0,062/2,24 ≈ 0,028 autour du ratio observé (~0,98) → borne basse plausible autour de
  0,95-0,97. **Je prédis que le critère est TENU DE JUSTESSE ou MANQUÉ DE PEU** — pas un verdict
  net dans un sens ou l'autre avant la mesure réelle. Falsificateur symétrique : si b et c
  restent aussi déséquilibrés qu'un artefact d'échantillon isolé (ex. un mauvais tirage sur une
  seule tâche fait chuter sa borne basse loin sous 97 % pendant que les 3 autres sont hauts),
  la MOYENNE peut encore satisfaire le critère malgré une tâche isolée faible — c'est le sens
  voulu par « en moyenne » (chef), pas une exigence tâche par tâche sur ce second critère
  (contrairement à McNemar, qui LUI est exigé sur CHAQUE tâche).

## Vérifications faites AVANT cette pièce (à sec, sans nouvelle mesure)

* `mcnemar-ic-237d.py` testé sur les jsonl DÉJÀ mesurés de la 261b (GSM8K n=50) : reproduit
  exactement les mêmes a/b/c/d/score/p que `mcnemar-261b.py`, plus l'IC bootstrap (borne basse
  0,9375 à R=5000) — aucune nouvelle mesure GPU, aucune carte engagée pour cette vérification.

## Ce que je NE fais PAS maintenant

Aucune mesure GPU dans cette pièce (ordre : « après 15 h, régime plein »). Ce scellé est écrit
et poussé à sec ; la mesure suit dans une pièce/branche reprise plus tard.
