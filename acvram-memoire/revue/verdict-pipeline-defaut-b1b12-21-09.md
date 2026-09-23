# Verdict — `test_pipeline_par_defaut_ids_au_bit_b1_et_b12` (Coder nvfp4, ACVRAM_PIPELINE=1 par défaut, 0.6.34) : **TENU 2/2** après correctif d'Océane (`oceane-pipeline-defaut` 030f0ab9, `seq.id`)

instrument : `tests/test_pipeline_decodage.py::test_pipeline_par_defaut_ids_au_bit_b1_et_b12[1,12]`, arbre pinné (`cd`) sur 030f0ab9 (0d77e742 + correctif test), un seul `carte.sh`, en-tête sans pair (charge 2,16, cinnamon/python/chromium)
scellé (Maîtresse, avant) : pipeline=1 rejoue le même graphe sur le même lot, ids identiques AU BIT contre le témoin pipeline=0 (b=1 et b=12) ; premier essai FAILED (`AttributeError: 'Sequence' object has no attribute 'sequence_id'`, ligne 177) — bug du test, pas du moteur (`GenerationOutput.sequence_id` existe, `Sequence.id` est le bon champ) ; corrigé, rejoué
mesuré : `2 passed in 57.11 s` — b=1 et b=12 identiques au bit entre pipeline actif et témoin
verdict : **TENU**
durée : essai 1 (échec de test) 23 s + essai 2 (correctif) 57 s ≈ 1,3 min ; HEAD 030f0ab9
suite : ma file : 4 alias k48/i8c/Ornith/Kimi sur main (0d77e742 fusionné avec le correctif) 0×500 → § 1b → GLM b=1

## Rejouable
`cd <worktree 030f0ab9> && ACVRAM_ARBRE=$PWD PYTHONPATH=$PWD ACVRAM_CPUS=0-15 ACVRAM_TYPE=mesure <venv>/bin/python -m pytest -q tests/test_pipeline_decodage.py::test_pipeline_par_defaut_ids_au_bit_b1_et_b12` sous `outils/carte.sh` (≤ 1 min).
