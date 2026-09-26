# Ids au bit b1/b12, main (runner.py scindé) — 21/09

* instrument : `tests/test_pipeline_decodage.py::test_pipeline_par_defaut_ids_au_bit_b1_et_b12`
* commit : 2d6d5fdb (worktree poste2-w-21-09, main fusionné, contient le refactor `runner.py` → `contexte.py`/`graphes.py`/`pipeline.py`, ancêtre 665eeacc)
* régime : ACVRAM_TYPE=mesure, sous carte.sh
* scellé : ids identiques bit à bit b1/b12
* mesuré : `2 passed in 118.88s`
* verdict : **TENU** — le refactor `runner.py` (déplacements purs annoncés) ne casse pas la reproductibilité bit à bit b1/b12 sur main. **Limite** : ce test compare b1 à b12 sur le MÊME arbre (main), pas les jetons produits par main contre ceux de 665eeacc côte à côte — aucun script de comparaison croisée entre deux arbres pour cette piste n'existe dans le dépôt à ma connaissance ; si une comparaison A/B explicite (665eeacc vs main) est voulue, il faut un script dédié (capture des jetons des deux côtés, diff), pas fait ici faute de temps/outil.
* durée : 122 s (16:19:52–16:21:54)

nvidia-smi propre avant/après (seul PID 4286).
