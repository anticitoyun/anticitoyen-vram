# Cellule b=12 Coder OFFICIELLE — TENU, 1 625,5 t/s, +1,84 % devant vLLM — 22/09 (Manon)

* instrument : `scratchpad/laurine-b12-21-09/cellule-officielle-22-09.sh` (harnais de service identique à la cellule 1 540 du 21/09 : `acvram.cli serve` + `banc-llamacpp-16-09.py decode`, fenêtre 20 s), worktree manon-w-21-09, main à jour
* commit : 2dfd7f4b ; scellé `revue/laurine-scelle-cellule-b12-22-09.md` — repli sur référence vLLM figée 1 596,1 t/s (vLLM non rejouée ce matin, faute d'instrument/temps sous la main)
* régime : AUCUNE variable d'environnement posée — confirmé sur la ligne de service : `sampler=graphe rapatriement=epingle graphes=on(hybrides≤12) eco=2700(2685)` (leviers 1+2 tous deux au défaut du code)
* scellé : prédit (Laurine) 1 590-1 660 t/s ; réfuté si médiane < 1 596,1 ; horloge par fenêtre, écart ≤ 3 %
* mesuré : f1 1603,0 · f2 1699,6 · f3 1632,0 · f4 1619,0 · f5 1597,9 · f6 1651,2 t/s — **médiane 1 625,5 t/s** ; horloge_sm_med 2677-2692 MHz (écart max 0,56 %, sous 3 %) ; les 6 fenêtres au-dessus de la référence 1 596,1
* verdict : **TENU** — médiane 1 625,5 dans la fourchette prédite [1 590 ; 1 660], **+1,84 % devant vLLM** (1 596,1). Cellule officielle du README.
* durée : ~5 min (chargement + 6 fenêtres de 20 s)

## Suite
`familles-noyaux.py --detail proj_etroites_int8` à sec sur la trace nsys du 22/09 (0 min de carte), puis PPL relative A/B 31B, C9, TRT-LLM.
