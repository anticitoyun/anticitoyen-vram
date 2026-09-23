# Cellule b=12 A/V vraie alternance — vLLM inexploitable (JSON tronqué), acvram confirmé — 22/09 (Manon)

* instrument : `scratchpad/laurine-b12-21-09/chaine-cellule-b12-vraie-vllm.sh` (A1 V1 A2 V2 A3 V3, chaque fenêtre prend son propre verrou), worktree manon-w-21-09
* commit : main 5d5d6f8e
* régime : acvram défauts (graphe+épinglé) ; vLLM `Qwen3-Coder-30B-A3B-Instruct-FP4-a16`, kv_cache_dtype=auto
* scellé : médiane acvram dans [1590 ; 1660] ; réfuté si médiane A < médiane V ; écart horloge par paire ≤ 3 %
* mesuré : **A** 1643,1 / 1622,9 / 1619,2 t/s (médiane **1622,9**, horloges 2557/2535/2565 MHz) — cohérent, dans la fourchette prédite. **V (vLLM) : les 3 fenêtres produisent un JSON TRONQUÉ**, coupé net après `"lots": 3, "prefill` sans valeur ni fermeture — aucun débit décodage récupérable sur aucune des 3 fenêtres (même défaut systématique, pas un accident isolé). Durée totale 12 min 34 s (06:10:41-06:23:15), bien sous les ~45 min prévues — cohérent avec des fenêtres V coupées avant la vraie mesure.
* verdict : **INDÉCIDABLE sur le critère A vs V** (aucun débit vLLM exploitable, le scellé « réfuté si médiane A < médiane V » ne peut pas être jugé) — **mais A seul confirme** le résultat du matin (médiane 1622,9, cohérent avec 1625,5 de `verdict-cellule-b12-officielle-22-09.md`, tous deux dans [1590;1660]). Cause du JSON tronqué hors mon domaine (harnais de mesure vLLM du script, `charge.py`/équivalent), à nommer par Laurine/Océane avant un nouvel essai.
* durée : 12 min 34 s de carte, verrou rendu propre entre chaque fenêtre

## Suite
Le repli publié ce matin (`verdict-cellule-b12-officielle-22-09.md`, TENU, 1625,5 t/s contre référence vLLM figée 1596,1) reste la cellule officielle valide — cette tentative de vraie alternance ne le remplace pas (vLLM inexploitable). M-UVA ensuite.
