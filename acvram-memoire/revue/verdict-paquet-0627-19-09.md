# Verdict — 0.6.27 (main 1874d664 ; GUI seule, serveur = 0.6.25 : le diff `acvram/` depuis c5367682 est la version et la variable de diagnostic `ACVRAM_MLA_QABS_DEUX_MOITIES`, opt-in) : **trois bras éco conformes ; `test_paquet_charge_utile` 4 passed ; ligne de régime `eco=2700(2692) glue=compact(8)`** — feu vert d'installation possible

instrument : `scratchpad/verif-eco-19-09/chaine.sh` (serve réel sans variable → régime + `acvram eco etat`, SIGTERM → libre ; `-lgc` à la main + `ACVRAM_ECO=off` → la ligne lit la carte ; requête servie dans chaque bras), 06:55-06:56, prises `carte.sh` ; test du paquet sur `acvram_0.6.27_amd64.deb` (975 Ko, `Version 0.6.27`) via copie temporaire du test (nom figé), hors carte ; `-rgc` rendu, 277 MHz au repos ; copies `scratchpad/cellule-0627-19-09/`
scellé (chef 06 h 54 / REGLES § 3, avant) : régime `eco=2700(...)` juste, `-rgc` à l'arrêt, pas de charge résiduelle, charge utile passée
mesuré : (1) réel : `eco=2700(2692)`, `etat {"mode": "2700", "pid": 55520}`, requête à 2 670 MHz, après SIGTERM `horloge=libre` (540-2 805, stable) ; (2) `-lgc` main + `ECO=off` : `eco=off(2670: verrou 2700 posé hors processus)`, après SIGTERM `lgc2700?` (posé hors acvram, non rendu : juste) ; `glue=compact(8)` présent ; `4 passed in 4.15s` ; carte vide après (seul 4284 = Qwen rapide sur la 3080 Ti)
verdict : **TENU** — même serveur que 0.6.25, mêmes trois lignes ; rien à corriger
suite : chef : feu vert d'installation 0.6.27 ; ma file : gemma:2 + coder:0 (9 tranches) → β 9 tranches → gemma capture

## Rejouable
`bash scratchpad/verif-eco-19-09/chaine.sh` (1,5 min) ; paquet : `sed 's|acvram_0.6.0_amd64.deb|acvram_0.6.27_amd64.deb|' tests/test_paquet_charge_utile.py > tests/tmp.py && CUDA_VISIBLE_DEVICES= <python> -m pytest tests/tmp.py -q ; rm tests/tmp.py`.
