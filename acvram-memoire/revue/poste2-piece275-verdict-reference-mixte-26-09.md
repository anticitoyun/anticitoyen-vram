# Pièce 275 — verdict référence `Qwen3.8-27B-unsloth-mixte-i8c` (poste2, 26/09) : PPL tenue, panel PLUS BAS que prédit sur 2 tâches

* **instrument** : `generer-reference-v2.sh` (PPL + 4 tâches, chacune sa propre `carte.sh`),
  `fusionner-panel.py`
* **commit** : `b42afb2f0` (prédiction, AVANT mesure) sur `poste2-275`
* **régime** : carte 0, `ACVRAM_ATTENTE=14400`, `ACVRAM_DUREE_MAX=3600` pour `professional_law`
  seule (1800 s insuffisant au 1er essai, tué à 98/150) ; les 4 autres pièces (PPL, gsm8k,
  high_school_mathematics, college_computer_science) déjà faites sous 1800 s, non reprises
* **scellé** : `revue/poste2-piece275-scelle-26-09.md`, addendum prédiction (avant mesure)
* **mesuré** :

| mesure | valeur | prédiction | tenue ? |
|---|---|---|---|
| PPL | **6,5048** | [6, 9] (large [4,15]) | **TENUE** |
| gsm8k (n=250) | 0,648 | au-dessus du hasard | tenue (largement) |
| mmlu high_school_mathematics (n=150) | **0,20** | au-dessus du hasard (25 %) | **NON TENUE** — sous le hasard |
| mmlu professional_law (n=150) | **0,08** | au-dessus du hasard (25 %) | **NON TENUE** — très sous le hasard |
| mmlu college_computer_science (n=100) | **0,11** | au-dessus du hasard (25 %) | **NON TENUE** — sous le hasard |
| moyenne panel | 0,2595 | — | — |

* **verdict** : **PPL tenue, panel NON tenu sur 3 des 4 tâches** — falsificateur pas déclenché
  au sens littéral du scellé (aucune tâche à 0 question valide), mais le résultat contredit
  clairement la prédiction qualitative (« au-dessus du hasard sur les 4 »). Trois tâches sur
  quatre tombent AU OU SOUS le niveau du hasard (25 % à 4 choix) alors que gsm8k reste correct
  (0,648) et la PPL brute est bonne (6,50, meilleure que le Coder à 9,27).
* **réserve, signalée sans être creusée (hors mandat de cette pièce)** : cette dissociation
  (bonne PPL + bon gsm8k, mais MMLU proche ou sous le hasard) ressemble aux deux bogues déjà
  rencontrés en 261b/275 (gabarit de conversation mal appliqué, ou coupe prématurée de la
  génération) — mais ceux-ci étaient déjà corrigés dans `panel-taches.sh`/`prise-tache-275.sh`
  et fonctionnent normalement sur le Coder (0,43 à 0,93 sur les mêmes tâches). Une hypothèse
  non vérifiée ici : ce modèle mixte répond différemment au gabarit de conversation par
  défaut, ou son format de réponse CoT diverge de ce que `get-answer` sait extraire — **pas
  mesuré, seulement noté**, comme demandé.
* **note demandée (chef), sur la lenteur croissante de `professional_law`** — À NOTER, PAS
  MESURÉ : les questions `professional_law` du split `test` MMLU sont BEAUCOUP plus longues
  que les autres sujets du panel (lues depuis le jeu de données HF, sans nouvelle mesure GPU) :
  moyenne **768 caractères** (max 2411) contre 289 pour `college_computer_science` et 143 pour
  `high_school_mathematics` — un facteur 2,7 à 5,4×. Cohérent avec l'hypothèse de chef (284,
  poste1) : sur un hybride à cache de préfixe, une invite qui dépasse la frontière d'un
  instantané fait retomber le préfill en série (deux passes) — des invites nettement plus
  longues expliquent naturellement le ralentissement croissant observé (15-20 s/item vers la
  fin, contre un rythme régulier sur les 3 autres tâches).
* **durée** : 4/5 pièces sous 1800 s chacune ; `professional_law` (reprise) sous 3600 s,
  terminée normalement cette fois
