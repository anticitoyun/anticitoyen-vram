# Verdict — -k48 en régime duel (prefill W4A16) : PPL 8,1569 = 1,0018, TENU (≤ 1,005) ; le -k48 prend le duel

- **instrument** : `scratchpad/reppl-eval-16-09.py` (ligne de poste2, wiki-gptq 2048/2048, min-context 256, 4 fenêtres, régime lu), `ACVRAM_MOE_MMA=0`, `ACVRAM_QA_COMPTE=1` ; sorties `scratchpad/ppl-k48-w4a16-16-09/`
- **commit** : travail/poste3 **d6c7b27** (code `acvram/` = main ce71723, noyau par ligne inclus) ; protocole même commit
- **régime** : prefill W4A16 (`chemin_moe=gemv/boucle`, **aucun bloc quantifié** : le compteur QA ne s'imprime pas, le W4A4 n'est pas pris), piles `oui` 46/46, 0 paramètre hors `cuda:0` ; deux passes identiques au 10⁻⁹, 6,6 / 6,4 s
- **scellé** (poste7 § 8.2) : ≤ 1,005 ; réfuté > 1,010 → `-avant-noawq-experts`. Moi : 1,000-1,004
- **mesuré** : **8,156925 / 8,1427 = 1,00175**
- **verdict** : **TENU** — `GLM-4.7-Flash-srcbf16-nvfp4-k48` est le converti du duel, régime « prefill W4A16, décodage MMA=1 MIN_T=5 » ; ma prédiction tenue (le retrait des 124 tables int8 coûte +0,004 sur le 0,998 du livrable, dans l'attendu ≤ 10 % d'poste1)

## Colonne PPL du duel (à reporter telle quelle)
```
converti -k48 (nvfp4 seul, AWQ experts alpha commun)   bf16 8,1427
  prefill W4A16 (régime duel)        8,1569   1,0018   tenu
  prefill W4A4 par ligne (877169c)   8,2913   1,0183   coût E2M1, chantier d'après-duel
```
Le décodage MMA=1 à MIN_T=5 n'est pas couvert par une PPL de prefill : à b=12 le godet 12 passe en MMA (W4A4), le coût réel au décodage reste à mesurer par la sortie du duel (ties/greedy) ou par une PPL de décodage (`ppl-narrow-b12`, adaptée à GLM) — à dire dans le verdict du duel, pas à cacher.

## Suite (ordre de chef)
Duel prise A dans ce régime : redémarrage de session avant le bloc, `git merge origin/main` avant la prise, scripts `campagne-duel-glm-15-09.sh` (acvram b=1/4/12 ×2 depuis le worktree avec `ACVRAM_MOE_MMA=0` au prefill, `ACVRAM_MODELE_MESURE=-k48`, prefill pp2048 à part) et bras vLLM GadflyII NVFP4 b=1/4/12 (2cb4e22).
