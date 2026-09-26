# Ids au bit b1/b12, sampler vectorisé — 21/09

* instrument : `tests/test_pipeline_decodage.py::test_pipeline_par_defaut_ids_au_bit_b1_et_b12` (fichier suivi)
* commit : d12b2a08 (worktree figé `poste2-d12b2a08-21-09`, `.venv` reconstruit)
* régime : ACVRAM_TYPE=mesure, sous `outils/carte.sh`
* scellé : ids identiques bit à bit entre b=1 et b=12 ; risque nommé d'avance (argmax bf16 sur égalité stricte, sampler vectorisé) — un id divergent = défaut, arrêt
* mesuré : `2 passed in 264.41s` (paramétré b=1, b=12)
* verdict : **TENU** — le sampler vectorisé (`d12b2a08`, glouton = argmax seul) ne casse pas la reproductibilité bit à bit entre les deux tailles de lot ; le risque nommé (argmax bf16 sur égalité stricte) ne s'est pas matérialisé.
* durée : 267 s (13:27:47–13:32:14), prévu ≤ 300 s (timeout), tenue

Verrou propre (carte.sh pris/rendu), nvidia-smi avant/après identique.
