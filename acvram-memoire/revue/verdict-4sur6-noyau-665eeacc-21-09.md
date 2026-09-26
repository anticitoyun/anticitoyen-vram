# test_nvfp4_4sur6.py -k noyau — 665eeacc — 21/09

* instrument : `tests/test_nvfp4_4sur6.py -k noyau`
* commit : 4e9c76ff (main, worktree poste2-w-21-09)
* régime : ACVRAM_TYPE=mesure, sous carte.sh
* scellé : test passe
* mesuré : `1 passed, 10 deselected in 159.19s`
* verdict : **TENU** — 1re tentative tuée à 60 s (mon timeout trop court, compilation JIT du noyau au premier lancement, cf. leçon des blocages précédents) ; relancée à 300 s, réussie en 159 s.
* durée : 161 s (16:16:16–16:18:57), verrou propre

nvidia-smi propre avant/après (seul PID 4286).
