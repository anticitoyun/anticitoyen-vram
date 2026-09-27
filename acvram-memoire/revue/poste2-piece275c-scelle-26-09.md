# Pièce 275c — scellé AVANT re-notation (poste2, 26/09, ordre chef) : nouveau filtre d'extraction, à sec, sans carte

## (1) Filtre existant réutilisable ?

Vérifié dans `mmlu/flan_cot_fewshot/_mmlu_flan_cot_fewshot_template_yaml` : une seule entrée
dans `filter_list`, `get-answer` — PAS de second filtre `flexible-extract` à réutiliser
(contrairement au gabarit `flan_cot_ZEROSHOT`, qui en a deux, `strict-match`+`flexible-extract`,
mais que la 261b a déjà écarté pour une autre raison — split `validation` trop petit). Une
nouvelle regex est donc nécessaire, écrite ici.

## Nouveau motif (à sec, sur les jsonl déjà écrits — aucune nouvelle mesure GPU)

```
(?i)answer[^A-D]{0,25}\(?([A-D])\)?
```
Toute occurrence du mot « answer » (insensible à la casse), puis jusqu'à 25 caractères qui NE
sont PAS une lettre A-D (ponctuation, « is », « :», retours à la ligne, gras Markdown, LaTeX),
puis la lettre isolée, avec ou sans parenthèses. **Dernière occurrence retenue** (une réponse
qui hésite puis conclut doit compter sa conclusion, pas une lettre citée en passant plus tôt).

## Prédiction (avant de lancer le script sur les 30 échantillons déjà classés en 275b)

Sur relecture manuelle (dans le verdict 275b, réserve sur la catégorie « b ») : la plupart des
« extrait faux » contenaient déjà la bonne lettre, noyée dans du texte. Je prédis que le
nouveau motif porte le score RÉEL au-dessus de 0,70 sur les trois tâches MMLU du mixte-i8c
(hors du hasard, cohérent avec gsm8k à 0,648 sur le même modèle) — PLUS HAUT que mon estimation
grossière de 275b (« ≈ 15-16/30 », soit 0,50-0,53) : en relisant précisément les fins de
réponse (pas seulement 120 caractères), plusieurs cas classés « c » (aucune lettre) en 275b
contiennent en fait la lettre juste sous une formule que je n'avais pas repérée en diagonale
(`\boxed{\text{(C) }}`, `Answer: (C) ...`). **Falsificateur** : si le score recalculé reste
sous 0,50, mon estimation initiale était la bonne et la relecture précise ne change rien
d'important.

## Test cassant prévu

`tests/test_extraction_275c.py` : sur les 30 échantillons déjà écrits, l'ANCIEN motif
(`get-answer` actuel) doit rendre exactement les scores officiels de la référence 275
(0,20 / 0,08 / 0,11) ; le NOUVEAU motif doit rendre un score ≥ 0,70 sur chaque tâche. Le test
doit ÉCHOUER si on lui fait rejouer l'ancien motif à la place du nouveau (bras cassant).

Écrit et poussé AVANT le calcul réel.
