# Verdict — 4 alias (k48, i8c, Ornith, Kimi-Linear) sur main 2919dd3c (0.6.34, arbre 0d77e742 fusionné) : **0 × 500 sur 4 TENU** ; **GLM k48 sert de nouveau à contexte plein 32768/32768** (OOM de graphes CUDA résolu, mieux que la prédiction 31744) ; **i8c reste sous le seuil demandé** : `ctx_tenu=11264 < 12288` — écart nommé

instrument : `scratchpad/s2-contexte-20-09/chaine.sh PRISE=4` (arbre pinné `poste2-qvl` = origin/main 2919dd3c, `cd "$L"` avant `python -m`), en-tête sans pair (charge 2,60, gpg/glab/python3)
scellé (chef, avant) : 0 × 500 sur les 4 ; k48 doit charger et tenir 31744 ; i8c ≥ 12288, sinon écart nommé
mesuré : **GLM k48** : TENU — `ctx_tenu` non atteint, sert à 32768 pleins (−64 → 200/32704/1/length 65,1 s ; +64 → 400 nommé) — **au-dessus de la prédiction (31744)**, la capture de graphes qui OOMait hier ne plante plus ; **Coder i8c** : DÉFAUT-CHAUFFE — `ctx_tenu=11264` (demandé 15360), **sous le seuil ≥ 12288 de 1 024 jetons (−8,3 %)** ; **Ornith** : TENU — 15296/15360 tenu (−64 → 200/15296/1/length 1,07 s) ; **Kimi-Linear** : DÉFAUT-CHAUFFE — `ctx_tenu=13312` (demandé 15360, sous 15360 mais aucun seuil minimal fixé pour cet alias)
verdict : **0 × 500 sur 4 : TENU** ; GLM k48 est un progrès net (chargeait avant à 0, sert maintenant à contexte plein) ; **i8c : écart nommé, 11264 < 12288** — à trancher (accepter le chiffre mesuré, ou creuser pourquoi i8c reste le plus serré des quatre malgré son ctx demandé le plus bas)
durée : 09:48:20-09:55:24 (7,1 min) ; HEAD `poste2-qvl` 2919dd3c
suite : chef : trancher l'écart i8c (11264 vs 12288) ; § 1b (commande exacte), GLM b=1 (charge < 1), NVTX poste4 (après 10 h 10) — non joués, la fenêtre se ferme ici

## Rejouable
`PRISE=4 ACVRAM_ARBRE=<arbre pinné> poste2=<poste2> PYA=<venv>/bin/python ACVRAM_MODELES=/mnt/AI_GENERATOR/models_acvram CPUS=0-15 bash <poste2>/scratchpad/s2-contexte-20-09/chaine.sh` (≈ 7 min).
