# lic — budget KV bf16 plafonné à 325 blocs : le plan budgète un jeton en int8 quel que soit le format (poste1, 01/10)

* instrument : `tests/test_kv_budget_format_lic.py` (processeur, `auto_plan` sur le profil 5090 + `_kv_blocks_per_device`, spec qwen32 du 146)
* commit : branche poste1-lic (sur main 3f10000bd)
* régime : à sec (`CUDA_VISIBLE_DEVICES=""`), formats int8, fp8_e4m3, bf16, fp16, k8v4, lm4 ; 10 240 jetons, un seul tenant
* scellé : prédiction S1 : 325 blocs = 10 240 × 8,125 / 16, attendue avant le test
* mesuré : avant correctif, bf16 et fp16 donnent **325 blocs = 5 200 jetons pour 10 240 planifiés**, et un bloc budgété à 33 280 o contre 65 536 réels (4 rouges, 8 verts). Après correctif, 12/12 ; 93 tests KV et planificateur verts
* verdict : **CAUSE NOMMÉE ET CORRIGÉE**. `tiering.py:471-476` calcule `kv_per_tok = spec.kv_bytes_per_token(opts.kv_bits = 8, fmt)`, et `config.py:319` ne suivait le format que pour lm4 (bits), k8v4 et MLA. `loader.py:1490-1494` découpe ensuite ce budget en blocs du format RÉEL (`KVCacheConfig(dtype=bf16)`), soit la moitié des blocs. fp8_e4m3 est juste (même largeur et mêmes échelles que int8). Correctif : `kv_bytes_per_token` prend 16 bits pour bf16 et fp16 ; le défaut servi (int8) ne bouge pas
* durée : 0 min de carte

## Reste
* Chemin sans spec (`_plan_from_manifest`, `spec is None`) : `kv_bytes_per_token` vient du manifeste (converti en int8). Sous
  ACVRAM_KV_FORMAT=bf16, il garderait le défaut. Hors des chargements servis (spec toujours passée), non corrigé ici.
* Sur carte : rejouer le bras S1 Devstral B/A en bf16 (poste6), après accord de chef et d'poste6. Prédit : invite de 7 865
  jetons admise, 640 blocs à 10 240.
