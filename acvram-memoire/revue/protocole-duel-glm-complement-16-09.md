# Protocole — duel GLM prise A, complément 2 (16/09) : régime duel, converti -k48, prédictions révisées

Laure, 16/09/2026, avant mesure ; complète `protocole-duel-mla-glm-15-09.md`
(9998eca) et son complément 2cb4e22. Ordre : Jérôme / Sage § 8 (ETAT.md 0e2fccb,
bloc 5). Arbre : travail/laure au commit de la prise (main fusionné juste avant,
nommé dans l'en-tête de l'unité).

## Ce qui change
- **Converti acvram : `GLM-4.7-Flash-srcbf16-nvfp4-k48`** (Manon, nvfp4 seul, AWQ
  experts alpha commun), PPL en régime duel **1,0018** (`verdict-ppl-k48-regime-duel`).
- **Régime nommé : prefill W4A16 (`ACVRAM_MOE_MMA=0`), décodage MMA=1, MIN_T=5**
  (`ACVRAM_MOE_DECODE_MMA=1`, `_MIN_T=5`) — exporté par la campagne, relu au JSON
  (`_MOE_DECODE_MMA_lu`, `_MIN_T_lu`) ; le prefill pp2048 acvram passe par le bras
  `w4a16` de `prefill-glm-acvram-15-09.py` (importe depuis `ACVRAM_ARBRE`).
- **Colonne PPL du verdict** : -k48 W4A16 1,0018 ; W4A4 par ligne 1,0183 (non
  pris au duel) ; vLLM GadflyII NVFP4 : PPL non mesurée ici (à dire).
- **Temps de prefill publié à part** (j/s pp2048, acvram w4a16 contre vLLM).
- Sorties : `scratchpad/duel-glm-16-09/`.

## Prédictions révisées (scellées) — la précédente est fausse d'un facteur 2,5
Ma prédiction du 15/09 (« nous : 450-600 t/s à b=12 ») ignorait la mesure de
Laurine : **-sansawq b=12 = 55,5 ms/pas, 197 t/s, 1,55 J/jeton** (glm-gateup, A1).
Je la retire et scelle :
1. **acvram -k48 b=12 : 54-58 ms/pas, 205-222 t/s, 1,45-1,60 J/jeton** (tables
   d'experts fusionnées dans le noyau ≤ 0,5 ms, MIN_T=5 sans effet à b=12 dans
   les rondes GLM… à vérifier : la queue passe en MMA W4A4 comme sur Coder,
   −1 à −2 %). b=4 : 30-36 ms ; b=1 : 22-27 ms (GEMV + MLA non fusionné).
2. **vLLM NVFP4 b=12 : 1,3-1,9 × notre débit** (270-420 t/s) — je maintiens mon
   bord contre Sage (0,7-1,3×) ; réfuté si ≤ 1,3×. b=1 : vLLM 1,0-1,5× nous.
3. vLLM fp8 dyn (source bf16) : OOM au chargement (31 Go de poids) — un résultat.
4. Prefill pp2048 : acvram w4a16 8 000-12 000 j/s (le m64e4 faisait 16 938 en
   W4A4 sur l'ancien converti ; W4A16 déquantifie) ; vLLM 6 000-10 000 j/s
   (TRITON_MLA num_stages=1). Réfuté si acvram < vLLM.
Ce qui n'est pas mesuré ici et se dit dans le verdict : la PPL au **décodage**
MMA W4A4 (godet 12) n'est pas la PPL de prefill ; le coût E2M1 y reste (Sage § 8.3).
