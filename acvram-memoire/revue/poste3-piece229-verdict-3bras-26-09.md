# Verdict — pièce 229, protocole à trois bras (poste3, 26/09, ordre chef) : la 209 seule porte la régression, TENU sur les deux comparaisons

* **instrument** : `banc-llamacpp-16-09.py` (invites réelles, 227), `scratchpad/poste3-p229-3bras-26-09/cellule-3bras.sh`, ABC répété ×5, processus séparés, serveur neuf par passe, garde de charge (load1 < 3) avant chaque lancement.
* **scellé** : `revue/poste3-piece229-scelle-3bras-26-09.md` (avant mesure).
* **bras** : A = be837ca1 · B = main au défaut (`ACVRAM_MARLIN_PAR_LIGNE=1`) · C = main avec `ACVRAM_MARLIN_PAR_LIGNE=0` forcé.
* **régime** : Qwen3-Coder-30B-A3B-nvfp4 pur, b=8, -lgc 2700 ; 5/5 passes par bras invalidées seulement par le bridage puissance (attendu, jamais par la charge).

## Mesuré (médianes sur 5)

| bras | débit | J/jeton net |
|---|---|---|
| A (be837ca1) | 1 750,0 t/s | 0,1139 |
| B (main défaut, 209 active) | 1 547,3 t/s | 0,1493 |
| C (main, 209 désactivée) | 1 784,0 t/s | 0,1114 |

* **B vs C (isole la 209 seule)** : C plus rapide de **+15,3 %** (1 784,0 contre 1 547,3), J −25,4 % (0,1114 contre 0,1493). Prédit +8 à +15 % : **TENU**, à la limite haute.
* **A vs C (isole tout SAUF la 209)** : C plus rapide de **+1,94 %** (1 784,0 contre 1 750,0), J −2,2 %. Prédit 0 à +8 % : **TENU**, proche de zéro.

## Verdict

**La 209 porte à elle seule l'essentiel de la régression** mesurée par mon banc (217, 229) sur Qwen3-Coder-30B-A3B-nvfp4 pur : B↔C explique −13 à −15 % de débit, A↔C reste presque neutre (+1,9 %). Les cinq autres pièces fusionnées entre be837ca1 et main (187, 194, 195b, 201, 212, 179) ne portent quasiment rien sur cet alias/protocole.

**Ceci contredit la 226 (poste6, +12,8 % débit / −18,4 % J sur PAR_LIGNE seul, `banc-chat-openai.py`)**, qui isolait exactement la même chose (PAR_LIGNE=0 vs 1) et trouvait le sens INVERSE. Les deux mesures sont propres (charge saine, invites réelles, processus séparés) : la différence vient du PROTOCOLE, pas d'un artefact —
`banc-chat-openai.py` (226) mesure UNE salve de b requêtes concurrentes à max_tokens, via `/v1/chat/completions` (gabarit de conversation) ; mon banc (217/229) mesure un débit SOUTENU, lots répétés pendant 20 s, via `/v1/completions` brut (invite réelle tokenisée, sans gabarit).

**Hypothèse à trancher** : le chemin marlin-w13 de la 209 a un coût (construction/état des piles, cache) qui s'amortit bien sur une salve courte mais se paie en continu sur un décodage soutenu — ou l'inverse selon l'angle. Pas encore mesuré ; piste pour la suite (nsys sur les DEUX protocoles, même alias, même commit).

* **durée** : scellé + mesure ≈ 20 min de carte (file derrière poste6-p228 avant le lancement).
