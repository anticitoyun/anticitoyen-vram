# Pièce 275b — verdict (poste2, 26/09) : catégorie dominante = (c), extraction — RÉFÉRENCE 275 (mixte-i8c) INVALIDE, le banc est en cause

* **instrument** : lecture directe des `samples_*.jsonl` déjà écrits (`--log_samples`), à sec,
  aucune carte, aucun nouveau calcul.
* **scellé** : `revue/poste2-piece275b-classement-scelle-26-09.md` (prédiction : (d) troncature
  `max_tokens` dominante, avant lecture).
* **mesuré** : 10 échantillons × 3 tâches (filtre `get-answer`), `max_gen_toks=1536` (le banc) ;
  longueur brute moyenne 1586 car. (hsm), 5164 (law), 3590 (ccs) — toutes largement sous la
  limite en jetons estimés (≈ 400/1290/900), aucune sortie visiblement coupée en plein mot.

| tâche | a (juste, extrait) | b (extrait, faux) | c (rien extrait) | d (tronqué) | e |
|---|---|---|---|---|---|
| high_school_mathematics | 3 | 5 | 1 | 1 | 0 |
| professional_law | 1 | 2 | 6 | 1 | 0 |
| college_computer_science | 1 | 2 | 4 | 3 | 0 |
| **total /30** | **5** | **9** | **11** | **5** | 0 |

* **verdict** : **(c) domine (11/30), ma prédiction (d) est FAUSSE** (5/30 seulement, et 2 de
  ces 5 sont eux-mêmes mal classés — voir réserve). **Falsificateur déclenché : la référence
  275 (mixte-i8c) est INVALIDE, le banc est en cause, pas le modèle.**
* **cause précise, lue directement dans les sorties (pas supposée)** : le filtre `get-answer`
  du gabarit `mmlu_flan_cot_fewshot` cherche littéralement `(?<=answer is )(.*)` (espace
  compris, sensible à la casse à ce niveau) — CE modèle conclut le plus souvent par
  `**Answer: (A)**`, `the correct answer is:\n**(C)...**`, ou une phrase sans "answer is "
  exact ("Therefore, only **I and II** are true. **Answer: (C) I and II only**") : aucun ne
  matche le motif, d'où `[invalid]`.
* **réserve sur la catégorie (b)** : en relisant les 9 cas « extrait, faux », **la plupart
  (7 sur 9, ex. `**(C) 50**` contre cible `(C)`) contiennent en fait la BONNE lettre** — le
  motif `(.*)` est glouton et capture toute la fin de phrase (la valeur numérique, la formule)
  en plus de la lettre, ce qui casse la comparaison stricte `exact_match`. Le vrai décompte de
  lettres correctes, tous cas confondus, avoisine 15-16/30 (bien au-dessus du hasard) une fois
  ce second défaut pris en compte — pas seulement les 5 comptés « a ».
* **conclusion** : la référence 275 du mixte-i8c (panel MMLU) est à REFAIRE après correction du
  banc — deux défauts cumulés dans `get-answer` : (1) motif d'extraction trop étroit pour un
  modèle qui varie sa formule de conclusion ; (2) capture gloutonne qui inclut du texte après
  la lettre, cassant `exact_match` même quand la lettre est juste. **Pas corrigé dans cette
  pièce (à sec, courte, sur ordre de chef)** — laissé à sa décision (pièce suivante ou
  correctif direct du banc).
