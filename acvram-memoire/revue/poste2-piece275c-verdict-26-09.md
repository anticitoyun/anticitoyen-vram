# Pièce 275c — verdict (poste2, 26/09) : nouveau motif d'extraction, panels avant/après, 237 confirmée touchée mais valable

* **instrument** : `scratchpad/poste2-p275-26-09/renoter-275c.py` (motif `(?i)answer[^A-D]{0,25}\(?([A-D])\)?`,
  dernière occurrence retenue), `tests/test_extraction_275c.py` (bras cassant : l'ANCIEN motif
  doit reproduire le score officiel, le NOUVEAU doit rendre ≥ 0,5 — vérifié à sec, script direct,
  `pytest` non lancé ici faute de `.venv` torch dans ce worktree, comme les pièces précédentes de
  cette chaîne — le fichier est prêt à être rejoué sous `carte.sh`).
* **commit** : `dd18896e5` (motif + scellé, AVANT re-notation) sur `poste2-275`
* **régime** : à sec, aucune carte, aucune nouvelle génération — relecture des jsonl déjà écrits
  (référence mixte-i8c ET référence Coder, les deux déjà en cache).

## (1) Filtre existant : aucun réutilisable

`mmlu/flan_cot_fewshot` n'a qu'un seul filtre (`get-answer`) — pas de second `flexible-extract`
à réutiliser (contrairement à `flan_cot_zeroshot`, écarté pour une autre raison, 261b). Nouveau
motif écrit (voir scellé).

## (2) Panels avant/après (ancien filtre officiel vs nouveau, mêmes sorties)

| modèle | tâche | ancien | nouveau | Δ |
|---|---|---|---|---|
| mixte-i8c | high_school_mathematics | 0,20 | **0,9067** | +0,71 |
| mixte-i8c | professional_law | 0,08 | **0,6733** | +0,59 |
| mixte-i8c | college_computer_science | 0,11 | **0,81** | +0,70 |
| Coder-30B | high_school_mathematics | 0,9267 | 0,9467 | +0,02 |
| Coder-30B | professional_law | 0,4333 | 0,6133 | +0,18 |
| Coder-30B | college_computer_science | 0,74 | 0,85 | +0,11 |

**Prédiction (275c) jugée** : hsm et ccs du mixte-i8c dépassent nettement 0,70 (TENUE) ;
professional_law du mixte-i8c atteint 0,6733, **sous mon seuil de 0,70 — prédiction NON TENUE
sur cette seule tâche**, bien qu'elle passe de 0,08 à 0,6733 (× 8,4), largement au-dessus du
hasard. Le test cassant confirme : ancien filtre reproduit EXACTEMENT les scores officiels sur
les 6 combinaisons (modèle × tâche), preuve que la nouvelle fonction et l'officielle mesurent
la même chose sous l'ancien motif — la divergence vient bien du motif, pas d'un bug de lecture.

## (3) La 237 (McNemar PAR_LIGNE) est-elle notée avec le même filtre ?

**Oui, confirmé** : les pièces 237b/c/d comparaient P0/P1 sur les MÊMES tâches
`mmlu_flan_cot_fewshot_{high_school_mathematics,professional_law,college_computer_science}`
avec le même filtre `get-answer`, déjà connu défaillant. Conséquence, comme chef le pose :
**la comparaison appariée (McNemar) reste VALABLE** — le motif est identique des deux côtés
(P0 et P1 tournent sous le MÊME banc), donc aucun biais directionnel n'est introduit entre les
deux bras ; MAIS de nombreuses réponses correctement formulées des deux côtés sont comptées
« fausses » par l'ancien filtre, ce qui **fait chuter le nombre effectif de paires
discriminantes** (beaucoup de paires deviennent artificiellement « les deux faux » — cellule d
de la table 2×2 — alors qu'elles seraient « les deux corrects » avec le bon motif). La
PUISSANCE statistique de 237d (capacité à détecter une vraie différence P1/P0 si elle existe)
est donc plus basse que son n nominal (150/150/150) ne le suggère — ses conclusions
(« aucune différence significative ») restent honnêtes mais MOINS assurées qu'annoncé.
**Non re-mesuré ici** (hors mandat, à sec) : si chef veut une 237 plus puissante, il faudrait
rejouer son McNemar avec le nouveau motif — pas fait dans cette pièce.

## Correctif proposé pour `outils/qualite.sh` / le banc (pas encore appliqué)

Remplacer le filtre `get-answer` par une regex config personnalisée (motif ci-dessus) via un
fichier de tâche `--include_path`, OU appliquer `renoter-275c.py` en post-traitement après
chaque `lm_eval run` (plus simple, aucune modification du gabarit lm-eval upstream). Laissé à
la décision de chef — pas appliqué dans cette pièce (à sec, courte, comme demandé).
