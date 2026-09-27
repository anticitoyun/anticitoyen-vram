# Pièce 237d — verdict (poste2, 26/09, ordre chef) : McNemar tenu partout, IC de récupération NON tenu — PAR_LIGNE reste à 0

* **instrument** : `prise-toutes-237d.sh` (8 prises, 2 bras × 4 tâches, chacune sous son propre
  `outils/carte.sh`), `mcnemar-ic-237d.py` (McNemar + IC bootstrap apparié, R=10 000, vérifié à
  sec sur les données de la 261b avant cette pièce)
* **commit** : `634df707f` (scellé + driver, AVANT mesure) sur `poste2-237d` (tirée de
  `poste2-261b`, fusionnée avec `origin/main` 0.7.0)
* **régime** : `Qwen3-Coder-30B-A3B-nvfp4`, `--max-batch 1`, carte 0, graine 1234 fixe (mêmes
  questions P0/P1 par tâche, vérifié par égalité des `doc_id`)
* **scellé** : `revue/poste2-piece237d-scelle-a-sec-26-09.md` (critère et prédiction écrits
  AVANT la mesure)
* **mesuré** :

| tâche | n | score P0 | score P1 | a | b | c | d | McNemar p | ratio P1/P0 | IC95% bas | IC95% haut |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 250 | 0,972 | 0,964 | 241 | 2 | 0 | 7 | 0,4795 | 0,9918 | **0,9793** | 1,0000 |
| mmlu high_school_mathematics | 150 | 0,920 | 0,9267 | 136 | 2 | 3 | 9 | 1,0 | 1,0072 | **0,9778** | 1,0388 |
| mmlu professional_law | 150 | 0,400 | 0,420 | 46 | 14 | 17 | 73 | 0,7194 | 1,0500 | **0,8814** | 1,2593 |
| mmlu college_computer_science | 100¹ | 0,760 | 0,750 | 69 | 7 | 6 | 18 | 1,0 | 0,9868 | **0,8961** | 1,0845 |

¹ n=100 et non 150 demandé : `college_computer_science` du split `test` MMLU ne contient QUE
100 questions — plafond du jeu de données, pas une limite de mon script (`--limit 150`
plafonne silencieusement à ce qui existe). Score identique au bit à celui de la 261b (0,76,
même modèle/graine/questions) — déterminisme confirmé, pas une coïncidence.

**Moyenne des 4 bornes basses : (0,9793 + 0,9778 + 0,8814 + 0,8961) / 4 = 0,9337 (93,4 %).**

* **verdict** :
  1. **McNemar p > 0,05 sur les 4 tâches** — critère 1 TENU (0,4795 / 1,0 / 0,7194 / 1,0,
     tous largement au-dessus de 0,05).
  2. **Borne basse moyenne de l'IC95% = 93,4 %, sous le seuil de 97 %** — **critère 2 NON
     TENU**. Deux tâches (`professional_law`, `college_computer_science`) tirent la moyenne
     vers le bas : leurs scores absolus sont plus modestes (40 % et 76 %) et leurs désaccords
     b/c plus nombreux (14/17 et 7/6) qu'en `gsm8k`/`high_school_mathematics` (quasi
     parfaits), ce qui élargit mécaniquement leur IC de ratio.
  3. **Conclusion (règle du scellé, appliquée telle qu'écrite) : PAR_LIGNE reste à 0 par
     défaut.** Le critère demandait les DEUX conditions ensemble ; la première est tenue, la
     seconde ne l'est pas — pas de preuve positive suffisante pour changer le défaut en 0.7.1.
* **prédiction (scellé) jugée** : j'avais prédit McNemar tenu partout (confirmé) et un critère
  IC « tenu de justesse ou manqué de peu » — le résultat (93,4 % contre 97 % requis, écart de
  3,6 points) correspond à « manqué », dans la fourchette que j'avais anticipée sans trancher
  à l'avance.
* **durée** : 8 prises, chacune sous son propre verrou, la plus longue (`professional_law`,
  150 questions à 1536 jetons max) ≈ 2 min — bien sous la borne 30 min ; total ≈ 15-20 min de
  carte cumulés, entrelacés avec d'autres files (régime plein)

## Ce qui resterait à faire pour trancher plus finement (hors mandat, pas fait ici)

Si chef veut retenter le critère IC à 97 % : augmenter n sur `professional_law` (déjà à son
plafond `test`, 150 = la taille réelle du split — pas d'augmentation possible sans mélanger
`validation`) et sur `college_computer_science` (plafonné à 100, idem) ne changera rien — ces
deux tâches sont déjà à la taille maximale de leur split `test`. Le seul levier restant serait
d'ajouter d'autres tâches MMLU (sujets différents) pour élargir l'échantillon global, ou
d'accepter un critère par tâche moins strict que 97 % en moyenne sur des sujets à score
intrinsèquement plus bas (40 % en droit professionnelle n'est pas un signe de mauvais
instrument — la 261b a déjà écarté cette hypothèse — juste une tâche plus dure).
