# Verdict — KV lm4 qualité (PPL de décodage, préfixe 8 192, Coder-30B) : lm4 / int8 = **1,0215**, réfuté (> 1,005 et > + 0,010) ; contrôles tenus (lm2 = 2,99×) ; capacité 1,97× par octets ; tq3+1 absent de l'arbre

instrument : `scratchpad/ppl-decode-kv-17-09.py` (préfixe 8 192 jetons de wiki-gptq entré par le prefill, puis 512 jetons notés un par pas en teacher forcing ; NLL par jeton dans chaque JSON ; mêmes 4 premiers ids `[14731, 284, 8397, 425]` dans les 4 bras) ; un processus par bras, chaîne `scratchpad/kv-lm4-17-09/chaine.sh`, JSON/logs `scratchpad/kv-lm4-17-09/`
commit : arbre laure 9ce76dd (= main 525895a + protocole `protocole-kv-lm4-qualite-17-09.md`, scellé avant la chaîne) ; `Qwen3-Coder-30B-A3B-nvfp4` ; torch 2.14.0+cu130
régime (`regime_ligne()`) : `[régime] ACVRAM_NARROW_GEMM=1 ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 ACVRAM_KV_FORMAT=<bras> extension=oui`, b=1, ctx 8 768 ; `cfg.dtype` effectif = le bras dans les 4 JSON ; **int8 seul en graphes** ; bf16, lm4, lm2 : « graphes CUDA désactivés, décodage en eager — capture impossible » (`graphs.paged_ok` exige int8, `graphs.py:299-300`) — la ligne `engine.regime_ligne()` dit pourtant `graphes=on` dans ces trois JSON : elle est lue à la création du moteur, avant l'échec de capture au premier pas (défaut d'instrument à corriger : `regime_ligne()` doit se relire après le premier pas)
scellé : int8 / bf16 < 0,002 · lm2 hors tolérance (> + 0,010) · lm4 / int8 ≤ 1,005 ET capacité ≥ 1,8 × int8 → régime opt-in ; Sage : 1,002-1,004 ; moi : 1,003-1,006, lm2 ≥ 1,03, capacité par octets 1,97, `tq3+1` absent (`kv_lm4.py:27`)
mesuré : bf16 **3,461** · int8 **3,4536** · lm4 **3,5279** · lm2 **10,3228** (512 notés chacun) → int8 / bf16 = 0,9979 (écart 0,0021, **au seuil**, int8 sous bf16 ; NLL apparié : moyenne + 0,0021 ± 0,0036, médiane 0, 58/512 jetons à |Δ| > 0,1 — non significatif, mais les deux bras diffèrent aussi par le chemin d'attention : noyau paginé int8 en graphes contre torch eager bf16) · lm2 / int8 = **2,989** (tenu : l'instrument voit le cache) · lm4 / int8 = **1,0215** (Δ NLL + 0,0213 ± 0,013, 161/512 jetons à |Δ| > 0,1, pire + 1,97 nats ; par quart de 128 : 1,012 / 1,032 / 1,013 / 1,029 — jamais sous 1,012) · capacité : blocs 4 384 / 2 428 = **1,806** (tenue de peu), octets 8 448 / 16 640 = **1,97** · `tq3+1` : `ValueError` (`tiering.py:50`), non mesuré
verdict : **lm4 réfuté comme défaut (scellé § 5 : + 0,0215 > + 0,010) et comme régime de capacité (1,0215 > 1,005) — quatre fois la tolérance, sur chaque quart de la tranche ; les deux prédictions (1,002-1,004 ; 1,003-1,006) sont fausses d'un ordre de grandeur. Les contrôles tiennent (lm2 casse à 2,99× ; int8 contre bf16 à 0,2 %, non significatif). La capacité 1,97× est réelle mais ne s'achète pas à ce prix ; `tq3+1` n'est pas dans l'arbre et n'est pas à mesurer (conditionné à lm4 ≤ 1,005). La conception se classe, comme Sage l'avait écrit pour cette issue.**

## Bras (PPL de décodage, 512 jetons après 8 192 de préfixe)

| bras | PPL | × int8 | × bf16 | graphes | blocs × 16 (capacité) | octets / bloc | `memory.used` MiB après chargement |
|---|---|---|---|---|---|---|---|
| bf16 | 3,4610 | 1,0021 | 1 | non (eager) | 19 728 | 32 768 | 23 489 |
| int8 (défaut) | 3,4536 | 1 | 0,9979 | oui | 38 848 | 16 640 | 23 807 |
| lm4 | 3,5279 | **1,0215** | 1,0193 | non (eager, `gather_fixed` → torch) | 70 144 | 8 448 | 22 445 |
| lm2 (témoin) | 10,3228 | 2,989 | 2,983 | non | 36 128 | 8 448 | 30 873 |
| tq3+1 | — | — | — | — | refus `ValueError: ACVRAM_KV_FORMAT='tq3+1' : attendu int8, fp8_e4m3, fp16, bf16, lm4, lm3 ou lm2` | — | — |

## Ce que les chiffres de capacité disent exactement
- Le rapport 1,806 en blocs n'est pas le rapport des formats : sous lm4 le plan s'est arrêté à `wanted = kv_per_tok × max_model_len × max_concurrent_seqs` (`tiering.py:456-458` : 8 768 × 8 = 70 144 jetons, exactement) alors que int8 et bf16 étaient bornés par le pool (40,4 Mo/couche tous deux). À pool égal, lm4 donnerait 40,4 Mo / 8 448 × 16 ≈ 76 500 jetons = 1,97 × int8 — le rapport des octets par bloc. Le critère ≥ 1,8 tient dans les deux lectures.
- `nvidia-smi memory.used` n'est pas un contrôle ici : 22,4 à 30,9 GiB selon le bras, avec des retries OOM de l'allocateur au chargement dans chaque log (le chargeur remplit la carte au bord) ; il mesure la réserve de l'allocateur, pas le cache. Le contrôle exact est `bytes_per_block × num_blocks`.
- lm2 a le même `bytes_per_block` que lm4 (8 448 : `width = 0.5 if rotated`, `kvcache.py:271`) mais un budget divisé par deux (`kv_lm4.bits`) : témoin seulement, sans conséquence ici.

## Réserves
- Une tranche, un modèle, 512 jetons : le rapport 1,0215 a une incertitude ± 0,013 (écart-type de la différence appariée / √512) — la borne basse 1,008 reste au-dessus de 1,005 ; sur chaque quart de 128 jetons le rapport est ≥ 1,012.
- Le préfixe est écrit par le chemin de prefill (torch `_quantize` pour lm4) et relu 512 fois : c'est bien le cache 4 bits que l'attention lit, à 8 k ; le bras lm2 le prouve. Une graine de rotation ou des centroïdes différents sont un autre format, pas un réglage de celui-ci (Sage, prémisse b).
- Aucune vitesse (chemin torch) — conforme à l'ordre.
