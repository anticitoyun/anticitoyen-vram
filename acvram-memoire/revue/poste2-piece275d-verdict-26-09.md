# Pièce 275d — verdict (poste2, 26/09) : correctif DANS le banc (config lm-eval dérivée, nom de tâche changé), tests passent, références mises à jour

* **instrument** : `outils/lm_eval_taches/mmlu_v275_{high_school_mathematics,professional_law,
  college_computer_science}.yaml` (dérivées de `mmlu_flan_cot_fewshot_*`, SEUL le `filter_list`
  change) ; `outils/panel-taches.sh`, `scratchpad/poste2-p275-26-09/{prise-tache-275.sh,
  generer-reference-v2.sh}`, `outils/qualite.sh` mis à jour pour les nouveaux noms +
  `--include_path outils/lm_eval_taches` ; `scratchpad/poste2-p275-26-09/mettre-a-jour-
  reference-v275.py` (re-notation sans GPU) ; `tests/test_lm_eval_taches_v275.py` (intégration,
  charge la VRAIE config) + `tests/test_extraction_275c.py` (bras cassant, mis à jour : les
  scores officiels historiques sont désormais fixés en dur, `panel.json` ne les porte plus par
  construction — le nom de tâche a changé, comme demandé).
* **commit** : `108c5b76b` (275c, base) sur `poste2-275`
* **régime** : à sec, aucune carte, aucune nouvelle génération.
* **scellé** : cette pièce n'avait pas de scellé séparé (correctif d'infrastructure sur un
  résultat déjà scellé et mesuré en 275b/c) — les prédictions restent celles de 275c.

## Le correctif vit dans le banc, pas en post-traitement

Nouveau `filter_list` (chaîne de 3 filtres lm-eval **enregistrés**, aucun code maison dans le
chemin de notation) :
```yaml
filter_list:
  - name: "get-answer-v275"
    filter:
      - function: "regex"
        regex_pattern: "(?i)answer[^A-D]{0,25}([A-D])"
        group_select: -1
      - function: "uppercase"
      - function: "map"
        mapping_dict: {"A": "(A)", "B": "(B)", "C": "(C)", "D": "(D)"}
        default_value: "[invalid]"
```
Nom de tâche changé (`mmlu_v275_*`, pas `mmlu_flan_cot_fewshot_*`) — le régime est porté par le
nom, aucun ancien score ne peut se mélanger à un nouveau dans un `panel.json` futur.

## Vérifications (aucune carte)

1. **`lm_eval validate --tasks mmlu_v275_*,... --include_path outils/lm_eval_taches`** :
   `All tasks found and valid`.
2. **`tests/test_lm_eval_taches_v275.py`** (charge la config depuis le FICHIER, instancie les
   classes `RegexFilter`/`UppercaseFilter`/`MapFilter` de `lm_eval.filters` — pas une
   réimplémentation) : la chaîne notera correctement les 30 sorties classées à la main
   (275b) et reproduit exactement les scores de 275c sur les 3 tâches complètes (150/150/100
   échantillons). **6/6 tests verts** (`.venv-panel/bin/pytest --noconftest`, ce worktree n'a
   pas le `.venv` torch du dépôt — à rejouer sous `pytest` normal + `carte.sh` pour la suite
   complète, comme les pièces précédentes de cette chaîne).
3. **`tests/test_extraction_275c.py`** (bras cassant conservé) : l'ancien filtre reproduit
   EXACTEMENT les scores historiques (0,20/0,08/0,11, fixés en dur car `panel.json` ne les
   porte plus) ; le nouveau motif reste ≥ 0,5.

## Références mises à jour (mêmes sorties, sans nouvelle génération)

| modèle | tâche | score (nouveau filtre, dans `panel.json`) |
|---|---|---|
| mixte-i8c | high_school_mathematics | 0,9067 |
| mixte-i8c | professional_law | 0,6733 |
| mixte-i8c | college_computer_science | 0,81 |
| mixte-i8c | gsm8k (inchangé) | 0,648 |
| **mixte-i8c** | **moyenne** | **0,7595** |
| Coder-30B | high_school_mathematics | 0,9467 |
| Coder-30B | professional_law | 0,6133 |
| Coder-30B | college_computer_science | 0,85 |
| Coder-30B | gsm8k (inchangé) | 0,92 |
| **Coder-30B** | **moyenne** | **0,8325** |

sha256 des `panel.json` mis à jour (fichiers hors git, `~/.cache/acvram/qualite-275/`) :
* mixte-i8c : `3c1a8a2d00487b93413cdbc75d14c238ef823c1328e690b0e0c6882da7b211f2`
* Coder-30B : `72cd9c681cad3274666a7e31f0d4ab51557774de174b22767c8eef63342b1367`

## Réponse à la question posée en 275c : la 237 est-elle affectée par cette version corrigée ?

**Non re-mesurée dans cette pièce** (à sec, sur ordre : correctif du banc + tests, pas une
nouvelle campagne McNemar). Ce qui est acquis : le correctif vit maintenant dans le banc sous
un nom de tâche neuf — toute future pièce (dont une éventuelle reprise de la 237/237d avec
`mmlu_v275_*`) mesurera juste, sans re-tomber dans le piège. La 237d elle-même reste ce qu'elle
était : une comparaison appariée valable mais moins puissante que son n nominal ne le suggère
(275c).

* **durée** : ~40 min à sec (écriture des 3 configs, du script de re-notation, des deux
  fichiers de test, exécution des tests, mise à jour des deux références).
