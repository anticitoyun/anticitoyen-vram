# Pièce 261b — verdict final (poste2, 26/09, ordre chef) : instrument étalonné, McNemar P1/P0 SANS différence significative

* **instrument** : `outils/panel-taches.sh` (261) corrigé en 3 temps (voir
  `revue/poste2-piece261b-etalonnage-scelle-26-09.md`) : `local-chat-completions` +
  `apply_chat_template`, `mmlu_flan_cot_zeroshot_*` puis `mmlu_flan_cot_fewshot_*` (split
  `test` 100 questions, pas `validation` 11), `max_gen_toks` 256→1536 ; `prise-tache-261b.sh`
  (une tâche, un moteur, une prise ≤ 30 min, reprise sur fichier déjà écrit) ;
  `mcnemar-261b.py` (paires par `doc_id`, filtre principal explicite)
* **commit** : `647e50b6d` (plancher de santé + instrument McNemar, AVANT P1) sur `poste2-261b`
  (tirée d'`origin/poste2-261`, fusionnée avec `origin/main` 0.7.0)
* **régime** : `Qwen3-Coder-30B-A3B-nvfp4`, `--max-batch 1`, carte 0, graine 1234 fixe (mêmes
  questions P0/P1, ordre déterministe lm-eval)
* **scellé** : `revue/poste2-piece261b-etalonnage-scelle-26-09.md` (addendum 2, plancher de
  santé écrit AVANT la mesure de santé, McNemar annoncé AVANT P1)
* **mesuré** :

| tâche | n | score P0 | score P1 | a (2 corrects) | b (P0 seul) | c (P1 seul) | d (2 faux) | McNemar (continuité) | p |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 50 | 0,98 | 0,96 | 48 | 1 | 0 | 1 | 0,0 | 1,0 |
| mmlu_flan_cot_fewshot_college_computer_science | 100 | 0,76 | 0,75 | 69 | 7 | 6 | 18 | 0,0 | 1,0 |

* **verdict** : **plancher de santé TENU** (GSM8K 0,98 ≥ 0,70 ; MMLU 0,76 ≥ 0,50) —
  instrument valide. **McNemar : AUCUNE différence significative entre P1 et P0 sur les deux
  tâches** (p = 1,0 des deux côtés — le nombre de désaccords b/c est quasi égal : 1/0 sur
  GSM8K, 7/6 sur MMLU, la correction de continuité écrase le peu de signal qu'il y a). Ceci
  confirme, avec un test statistique correct cette fois (contrairement à la 237c, dont l'IC
  de Wilson n'était de toute façon pas le bon outil pour une comparaison appariée), ce que la
  237b (KL quasi égales) et la 237c (IC recouvrants) suggéraient déjà sans le prouver :
  **PAR_LIGNE=1 ne dégrade pas la qualité par rapport à PAR_LIGNE=0 sur ces deux tâches, à cet
  effectif.**
* **réserve honnête** : n reste modeste (50 et 100) — un effet FIN (quelques points de
  recouvrement) resterait indétectable ; McNemar est bien plus puissant qu'une comparaison de
  proportions indépendantes à n égal (il neutralise la difficulté de la question, ne compte
  que les CHANGEMENTS de verdict), mais pas magique. Une seule tâche MMLU couverte
  (`college_computer_science`) — les deux autres (`professional_law`,
  `high_school_mathematics`) non reprises dans cette pièce (hors mandat : chef n'a demandé
  qu'UNE tâche MMLU pour le plancher, pas les trois pour McNemar).
* **durée** : plancher P0 (2 prises) + P1 (2 prises), chacune < 5 min de carte (modèle déjà en
  cache disque, chargement rapide), largement sous la borne 30 min

## Récapitulatif de la chaîne 261 → 261b → 237c/d

1. **261** : instrument livré, 5-shot brut sans gabarit — jamais mesuré en vrai avant la 237c.
2. **237c** : MMLU au hasard (18-33 %), GSM8K 0/50 — instrument faux, pas le modèle (diagnostic
   de chef).
3. **261b** : 3 bogues trouvés et corrigés (gabarit de conversation, tâche CoT mal choisie,
   longueur de génération) ; plancher de santé défini et TENU ; McNemar P1/P0 sans différence
   significative sur 2 tâches.

## Suite proposée (pas faite ici, hors mandat)

Étendre McNemar aux deux autres tâches MMLU (`professional_law`, `high_school_mathematics`)
si chef veut plus de couverture avant la décision finale sur le retour de `PAR_LIGNE=1` en
0.7.1.
