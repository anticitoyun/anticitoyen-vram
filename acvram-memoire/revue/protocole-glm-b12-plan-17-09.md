# Protocole — GLM `-k48-calibA` b=12 sous `bf16` avec budget KV pour 12 séquences (`CERT_PLAN_LEN=3072`)
instrument : `certifie-b12-15-09.py` rondes ×2 (ctx 2048, invite 256, 20 s) avec `CERT_PLAN_LEN=3072` (plan = 2 048 × 12 / 8, `tiering.py:458`) ; preuve : aucune ligne « budget KV épuisé », `n_jetons = n_pas × 12`.
commit : arbre laure 0edc3b9 (= main) ; régime classé `ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 ACVRAM_NARROW_GEMM=1`, `ACVRAM_PREFILL` défaut bf16, `ACVRAM_PREFILL_GROUPED` défaut (bmm — décodage non concerné, prefill des 12 × 256 concerné : il coûte plus, pas moins).
scellé (Sage) : 548-560 t/s. Moi : 545-555 t/s (Coder + 1,8 % ; GLM MLA : l'attention ∝ ctx pèse moins qu'en GQA int8, le prefill bmm plus lent coûte ~0,3 %), 0,72-0,74 J.
falsification : < 540 → le budget ne changeait rien pour GLM (ou bmm coûte plus que prévu) ; > 565 → la traîne pesait plus sur GLM que sur Coder.
