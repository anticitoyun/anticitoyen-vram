# Levier 2 (1) : ids/logprobs au bit b=1/b=12 sous ACVRAM_RAPATRIEMENT_EPINGLE=1 — TENU — 22/09 (poste2)

* instrument : `outils/carte.sh pytest tests/test_pipeline_decodage.py -q` (worktree poste2-w-21-09, main d7e57da2+)
* commit : main à jour (levier 2 7f967671 fusionné, défaut `sampler=graphe` depuis 13de908e)
* régime : `ACVRAM_RAPATRIEMENT_EPINGLE=1` (au-dessus du défaut graphe), `sampler=graphe`
* scellé : ids ET logprobs au bit, b=1 et b=12
* mesuré : `test_pipeline_par_defaut_ids_au_bit_b1_et_b12[1]` PASSED, `[12]` PASSED, ET `test_pipeline_bit_identique_a_egalite_pres` PASSED (le 3e test, cassé ce matin par le même défaut `depuis_graphe`, confirme le correctif d'instrument poussé plus tôt)
* verdict : **TENU**, 3/3
* durée : 85,1 s

## Suite
Frontière (2) : `frontiere-pas.py` A (graphe seul) / B (graphe + épinglé), 300 pas, seuil trou_gpu ≥30µs de baisse supplémentaire (réfuté si >100µs restant).
